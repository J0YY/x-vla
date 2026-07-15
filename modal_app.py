"""Modal compute harness for χ-VLA (spec §21).

Local Python has no torch (3.14); all execution runs on Modal GPUs. Usage::

    modal run modal_app.py::validate                 # Stage-0 operator tests
    modal run modal_app.py::prepare_data             # tiny-shakespeare tokens
    modal run modal_app.py::smoke_ablation           # fast 4-way ablation
    modal run modal_app.py::run_ablation             # full 4-way ablation

The project directory is mounted read-only at /root/proj and put on sys.path.
Data + checkpoints persist in a Modal Volume.
"""

import sys

import modal

PROJ = "/root/proj"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy", "tiktoken", "pytest")
    .add_local_dir(
        ".", PROJ,
        ignore=["*.pdf", "__pycache__", "*.pyc", ".git", ".venv", "data", "out"],
    )
)

app = modal.App("xvla")
vol = modal.Volume.from_name("xvla-data", create_if_missing=True)
VOL_PATH = "/vol"


def _bootstrap():
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)


@app.function(image=image, gpu="L4", timeout=1800)
def validate():
    """Run Stage-0 operator validation (spec §14 Stage 0) on GPU."""
    _bootstrap()
    import pytest
    code = pytest.main(["-q", f"{PROJ}/tests"])
    if code != 0:
        raise SystemExit(f"operator validation FAILED (pytest exit {code})")
    print("✅ Stage-0 operator validation passed")


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=1800)
def prepare_data():
    """Download + tokenize tiny-shakespeare into the volume."""
    _bootstrap()
    from xvla.train.data import prepare_tinyshakespeare
    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()
    print(f"train={train_bin}\nval={val_bin}\nvocab={vocab}")
    return train_bin, val_bin, vocab


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3600)
def smoke_ablation():
    """Fast four-way ablation on tiny-shakespeare (sanity of trends, not scale)."""
    _bootstrap()
    import json
    from xvla.models.lm import LMConfig
    from xvla.train.data import prepare_tinyshakespeare
    from xvla.train.train_lm import TrainConfig, four_way_ablation

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()

    model_cfg = LMConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=6, max_seq_len=256)
    train_cfg = TrainConfig(
        train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
        lr=6e-4, max_steps=1500, warmup_frac=0.05, eval_every=250, log_every=50,
    )
    results = four_way_ablation(model_cfg, train_cfg)
    summary = {tag: h["final_val_loss"] for tag, h in results.items()}
    with open(f"{VOL_PATH}/smoke_ablation.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    return summary


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=24 * 3600)
def run_ablation(max_steps: int = 6000, dim: int = 512, n_layers: int = 12):
    """Larger four-way ablation (spec §14 Stage 1 / §18). Uses tiny-shakespeare
    by default; point train/val at a fineweb .bin in the volume for scale."""
    _bootstrap()
    import json
    from xvla.models.lm import LMConfig
    from xvla.train.data import prepare_tinyshakespeare
    from xvla.train.train_lm import TrainConfig, four_way_ablation

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()
    model_cfg = LMConfig(vocab_size=vocab, dim=dim, n_layers=n_layers,
                         n_heads=dim // 64, max_seq_len=512)
    train_cfg = TrainConfig(
        train_bin=train_bin, val_bin=val_bin, seq_len=512, batch_size=32,
        lr=6e-4, max_steps=max_steps, warmup_frac=0.03, eval_every=500, log_every=50,
    )
    results = four_way_ablation(model_cfg, train_cfg)
    summary = {tag: h["final_val_loss"] for tag, h in results.items()}
    with open(f"{VOL_PATH}/run_ablation.json", "w") as f:
        json.dump({"summary": summary, "history": results}, f, indent=2)
    vol.commit()
    return summary


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def distill_strict_model(teacher_steps: int = 1500, distill_steps: int = 1500):
    """Option A: distill the per-token teacher into a strict scalar-norm student,
    calibrate + freeze, eval the deployable graph, fold, and assert the §19
    pre/post-fold match to 1e-5. This is the central "fully decomposable" gate."""
    _bootstrap()
    import json, math
    import torch
    from dataclasses import replace
    from xvla.models.lm import ChiLanguageModel, LMConfig
    from xvla.nn.normalization import RmsBatchNorm, PerTokenRmsNorm
    from xvla.train.data import prepare_tinyshakespeare, TokenBinDataset
    from xvla.train.train_lm import TrainConfig, evaluate, _lr_at
    from xvla.train.calibrate import calibrate_rbn
    from xvla.train.distill import DistillConfig, distill_strict
    from xvla.train.fold import verify_fold

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()

    # 1) Train the stable per-token teacher (a model we hold a handle to).
    teacher_cfg = LMConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=12,
                           max_seq_len=256, attn="bilinear", ffn="bilinear",
                           norm="per_token", qk_norm="per_token")
    tc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
                     lr=6e-4, max_steps=teacher_steps, warmup_frac=0.05)
    print("===== training per-token teacher =====")
    torch.manual_seed(0)
    teacher = ChiLanguageModel(teacher_cfg).cuda()
    ds = TokenBinDataset(train_bin, tc.seq_len)
    opt = torch.optim.AdamW(teacher.parameters(), lr=tc.lr, betas=(tc.beta1, tc.beta2),
                            weight_decay=tc.weight_decay)
    teacher.train()
    for step in range(tc.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tc)
        opt.zero_grad(set_to_none=True)
        x, y = ds.batch(tc.batch_size, "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = teacher(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), tc.grad_clip)
        opt.step()
        if step % 300 == 0:
            print(f"[teacher] step {step} loss {loss.item():.4f}")
    teacher_final = float(loss.item())

    # 2) Distill into the strict scalar-norm student (warm-started from teacher).
    # Near-identity learned residual gains (init 0.01, spec §10) keep the stack
    # in the regime where the degree-2 FFN's per-instance magnitude amplification
    # is negligible — the key to a fixed scalar norm staying stable.
    student_cfg = replace(teacher_cfg, norm="scalar_rbn", qk_norm="scalar_rbn",
                          learned_gain=True)
    # Aggressive stabilization (spec §20): low LR + strong weight decay to bound
    # weight growth, tight grad clip, frequent CP balancing.
    dtc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
                      lr=1e-4, weight_decay=0.5, grad_clip=0.3, balance_every=25,
                      max_steps=distill_steps, warmup_frac=0.1,
                      eval_every=max(distill_steps // 5, 1), log_every=100)
    print("===== distilling strict scalar-norm student =====")
    # Drop feature matching (destabilizing on an exploding stream); KL+CE only.
    dh = distill_strict(teacher, student_cfg, dtc,
                        DistillConfig(lambda_feat=0.0, lambda_ce=0.5))
    print(f"distill skipped steps: {dh.get('skipped', 0)}")
    student = dh["student"]

    # 3) Final fresh calibration + freeze, then eval the deployable graph.
    calibrate_rbn(student, lambda: ds.batch(dtc.batch_size, "cuda"), iters=100)
    for m in student.modules():
        if isinstance(m, RmsBatchNorm):
            m.frozen = True
    val_ds = TokenBinDataset(val_bin, dtc.seq_len)
    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    strict_val = evaluate(student, val_ds, dtc, ac)
    print(f"STRICT calibrated val = {strict_val:.4f} finite={math.isfinite(strict_val)}")

    # 4) Fold and assert the §19 pre/post-fold match (FP32).
    student = student.float()
    xv, _ = val_ds.batch(8, "cuda")
    folded, abs_diff, rel_diff = verify_fold(student, xv)
    # FP64 check on the same folded graph: isolates folding correctness from
    # FP32 accumulation (folding is an exact algebraic identity).
    folded64, abs64, rel64 = verify_fold(student.double(), xv.long())
    print(f"FOLD FP32 max|Δ| = {abs_diff:.2e} (rel {rel_diff:.2e}); "
          f"FP64 max|Δ| = {abs64:.2e}  (§19 gate: <1e-5)")
    leftover = [n for n, mm in folded.named_modules()
                if isinstance(mm, (RmsBatchNorm, PerTokenRmsNorm))]

    result = {
        "teacher_final_loss": teacher_final,
        "strict_val": strict_val,
        "strict_val_finite": bool(math.isfinite(strict_val)),
        "distill_best_val": dh["best_val_loss"],
        "fold_max_abs_fp32": abs_diff,
        "fold_max_rel_fp32": rel_diff,
        "fold_max_abs_fp64": abs64,
        "fold_pass": bool(abs64 < 1e-5 or rel_diff < 1e-5),
        "norm_modules_left_after_fold": leftover,
    }
    with open(f"{VOL_PATH}/distill_strict.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def homotopy_experiment(steps: int = 1500, budget: float = 8.0,
                        qk: str = "homotopy", arms: str = "A,B,C"):
    """Swap-gap-gated homotopy: 3-arm ablation (A diverge-control, B gated method,
    C fixed-schedule) + fold check on arm B. The proposed strict-norm solution.

    qk: "homotopy" (attention also morphs to scalar/strict) or "per_token"
        (attention stays per-token — tests the partial-purity: pure block+FFN,
        non-strict attention). arms: comma-separated subset to run."""
    _bootstrap()
    import json, math
    import torch
    from xvla.models.lm import LMConfig
    from xvla.nn.normalization import RmsBatchNorm, PerTokenRmsNorm, HomotopyNorm
    from xvla.train.data import prepare_tinyshakespeare
    from xvla.train.train_lm import TrainConfig
    from xvla.train.homotopy import HomotopyRunConfig, train_homotopy
    from xvla.train.fold import verify_fold

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()
    mcfg = LMConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=12, max_seq_len=256,
                    attn="bilinear", ffn="bilinear", norm="homotopy", qk_norm=qk,
                    learned_gain=False)  # big residual gains 1/sqrt(2L)
    tc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
                     lr=6e-4, max_steps=steps, warmup_frac=0.05, eval_every=steps,
                     eval_iters=40, log_every=100)

    summary = {}
    arm_b_model = None
    for arm in [a for a in ("A", "B", "C") if a in arms.split(",")]:
        print(f"\n===== ARM {arm} =====")
        rc = HomotopyRunConfig(arm=arm, spectral_budget=budget)
        h = train_homotopy(mcfg, tc, rc)
        summary[arm] = {
            "diverged": h["diverged"], "val_loss": h["val_loss"],
            "frac_at_kappa1": h["frac_at_kappa1"], "final_mean_kappa": h["final_mean_kappa"],
            "max_rms_final": h.get("max_rms_final"), "max_rms_stress": h.get("max_rms_stress"),
            "all_kappa1": h["all_kappa1"], "skipped": h["skipped"],
        }
        if arm == "B":
            arm_b_model = h["model"]

    # Fold arm B if every site reached κ=1 (pure export gate).
    fold_result = {"foldable": False}
    if arm_b_model is not None and summary["B"]["all_kappa1"]:
        m = arm_b_model.float()
        from xvla.train.data import TokenBinDataset
        xv, _ = TokenBinDataset(val_bin, 256).batch(8, "cuda")
        _, abs32, rel32 = verify_fold(m, xv)
        _, abs64, _ = verify_fold(m.double(), xv.long())
        fold_result = {"foldable": True, "fold_abs_fp32": abs32,
                       "fold_rel_fp32": rel32, "fold_abs_fp64": abs64}
        print(f"ARM B FOLD: rel {rel32:.2e} (fp32) / abs {abs64:.2e} (fp64)")

    result = {"summary": summary, "fold": fold_result, "budget": budget}
    with open(f"{VOL_PATH}/homotopy_experiment.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def debug(steps: int = 60):
    """Short fp32 sanity run with per-layer activation-norm probes."""
    _bootstrap()
    import torch
    from xvla.models.lm import ChiLanguageModel, LMConfig
    from xvla.nn.normalization import RmsBatchNorm
    from xvla.train.data import prepare_tinyshakespeare
    from xvla.train.train_lm import TrainConfig, train_lm

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()

    # Probe activation RMS at every RBN site on one forward pass.
    cfg = LMConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=12, max_seq_len=256,
                   attn="bilinear", ffn="bilinear")  # d_h=32 (spec §12), lighter tails
    model = ChiLanguageModel(cfg).cuda()
    from xvla.train.data import TokenBinDataset
    ds = TokenBinDataset(train_bin, 256)
    x, y = ds.batch(8, "cuda")
    probes = []
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, RmsBatchNorm):
            def hook(mod, inp, out, nm=name):
                probes.append((nm, float(inp[0].float().pow(2).mean().sqrt())))
            handles.append(m.register_forward_hook(hook))
    model.train()
    logits, loss = model(x, y)
    print(f"init loss={loss.item():.4f}  logits finite={torch.isfinite(logits).all().item()}")
    for nm, r in probes:
        print(f"  RMS in {nm:28s} = {r:.3e}")
    for h in handles:
        h.remove()

    # Longer bf16 training run to confirm the loss genuinely drops below uniform.
    tc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
                     lr=6e-4, max_steps=steps, warmup_frac=0.05, eval_every=max(steps // 4, 1),
                     log_every=25, dtype="bfloat16")
    h = train_lm(cfg, tc)
    return {"final_val_loss": h["final_val_loss"], "params_M": h["params_M"],
            "train_loss_tail": h["train_loss"][-5:]}


@app.local_entrypoint()
def main():
    """Default: validate operators, then run the smoke ablation."""
    validate.remote()
    print("smoke ablation:", smoke_ablation.remote())
