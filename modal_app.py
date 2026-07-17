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
    .pip_install("torch>=2.2", "torchvision", "numpy", "scipy", "tiktoken", "pytest",
                 "datasets", "huggingface_hub", "pillow")
    .add_local_dir(
        ".", PROJ,
        ignore=["*.pdf", "__pycache__", "*.pyc", ".git", ".venv", "data", "out"],
    )
)

app = modal.App("xvla")
vol = modal.Volume.from_name("xvla-data", create_if_missing=True)
VOL_PATH = "/vol"

# ---- LIBERO closed-loop simulator image (MuJoCo + robosuite + LIBERO, headless EGL) ----
libero_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-dev", "libglib2.0-0", "libosmesa6-dev",
                 "libegl1-mesa-dev", "libgles2-mesa-dev", "libglfw3", "libglew-dev",
                 "patchelf", "gcc", "g++")
    .pip_install("torch>=2.2", "torchvision", "numpy<2", "scipy", "pillow",
                 "huggingface_hub", "datasets", "opencv-python-headless",
                 "mujoco==3.1.6", "robosuite==1.4.1", "bddl", "easydict",
                 "termcolor", "thop", "cloudpickle", "gym==0.25.2", "hydra-core", "pyyaml",
                 "matplotlib", "imageio", "imageio-ffmpeg", "future")
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "cd /opt/LIBERO && pip install --no-deps -e .",
        # pre-write LIBERO's config so first import doesn't prompt interactively (EOFError)
        "mkdir -p /root/.libero",
        "python3 -c \"import os,yaml; root='/opt/LIBERO'; f=lambda n:next((d for d,s,fs in os.walk(root) if os.path.basename(d)==n),''); b=f('bddl_files'); i=f('init_files'); a=f('assets'); yaml.safe_dump({'benchmark_root':os.path.dirname(b),'bddl_files':b,'init_states':i,'assets':a,'datasets':root+'/datasets'}, open('/root/.libero/config.yaml','w')); print(open('/root/.libero/config.yaml').read())\"")
    .env({"PYTHONPATH": "/opt/LIBERO",
          "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", PROJ,
                   ignore=["*.pdf", "__pycache__", "*.pyc", ".git", ".venv", "data", "out"])
)


@app.function(image=libero_image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def libero_rollout(steps: int = 6000, n_frames: int = 40000, horizon: int = 8, res: int = 64,
                   eps_per_task: int = 10, max_task: int = 10, exec_h: int = 8):
    """CLOSED-LOOP eval (Q0): train the χ-VLA offline on LIBERO-Object (lerobot data,
    exact M4b pipeline) then roll it out in the MuJoCo/robosuite sim and report task
    success rate. Reconstructs the sim obs to match the training convention (agentview
    flipped+resized+/255; state = eef_pos + eef axis-angle + gripper_qpos; actions
    un-normalized) with a state-distribution sanity check before rollout."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig
    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"

    # ---- tasks + vocab (same as train_vla_libero) ----
    tasks = {}
    for tf in [f for f in list_repo_files(name, repo_type="dataset")
               if "task" in f.lower() and f.endswith((".jsonl", ".json", ".parquet"))]:
        try:
            tp = hf_hub_download(name, tf, repo_type="dataset")
            if tp.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(tp).reset_index()
                tcol = "task" if "task" in df.columns else next(c for c in df.columns if df[c].dtype == object)
                icol = "task_index" if "task_index" in df.columns else ("index" if "index" in df.columns else df.columns[0])
                for _, r in df.iterrows(): tasks[int(r[icol])] = str(r[tcol])
            else:
                for line in open(tp): r = json.loads(line); tasks[int(r["task_index"])] = r["task"]
            if tasks: break
        except Exception as e: print(f"{tf}: {e}")
    words = set()
    for t in tasks.values(): words.update(t.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words): vocab[w] = len(vocab)
    T = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T] + [0] * max(0, T - len(ids)))[:T]

    # ---- frames (cached) ----
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    if os.path.exists(cache):
        frames = pickle.load(open(cache, "rb"))
    else:
        from PIL import Image
        ds = load_dataset(name, split="train", streaming=True); frames = []
        for ex in ds:
            img = ex["observation.images.image"]
            if not isinstance(img, Image.Image): img = Image.fromarray(np.array(img))
            frames.append((int(ex["episode_index"]), int(ex["frame_index"]),
                           np.asarray(img.resize((res, res)), dtype=np.uint8),
                           np.asarray(ex["observation.state"], dtype=np.float32),
                           np.asarray(ex["action"], dtype=np.float32), int(ex["task_index"])))
            if len(frames) >= n_frames: break
        pickle.dump(frames, open(cache, "wb")); vol.commit()
    print(f"{len(frames)} frames")
    from collections import defaultdict
    eps = defaultdict(list)
    for f in frames: eps[f[0]].append(f)
    samples = []
    for ep, fs in eps.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"state_dim={state_dim} action_dim={d_a}; train state mean={np.round(s_mu,3)}")

    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a)
    model = ChiVLA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=8e-4, max_steps=steps, warmup_frac=0.05)
    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
        idx = torch.randint(len(samples), (256,), device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs[idx], instr[idx], states[idx],
                            torch.zeros(256, dtype=torch.long, device=dev), target_actions=actions[idx])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0: print(f"  step {step} mse {loss.item():.4f}")
    model.eval()

    # ---- rollout in sim ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    @torch.no_grad()
    def act(obs, instr_ids):
        raw = build_state(obs)
        img = np.asarray(Image.fromarray(obs["agentview_image"][::-1]).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(raw) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()          # (H, d_a) un-normalized

    n_tasks = min(max_task, suite.n_tasks)
    per_task, sane = {}, None
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        succ = 0
        for ep in range(eps_per_task):
            env.seed(ti * 100 + ep); obs = env.reset()
            if sane is None:                                   # state-match sanity (once)
                sane = {"sim_state": np.round(build_state(obs), 3).tolist(),
                        "train_state_mean": np.round(s_mu, 3).tolist()}
                print("STATE SANITY:", sane)
            done = False
            for t in range(220):
                chunk = act(obs, instr_ids)
                for k in range(min(exec_h, H)):
                    obs, r, done, info = env.step(chunk[k].tolist())
                    if done: break
                if done: break
            succ += int(bool(done))
        env.close()
        per_task[task.language] = round(succ / eps_per_task, 3)
        print(f"[task {ti}] {task.language}: {succ}/{eps_per_task}")
    overall = round(float(np.mean(list(per_task.values()))), 3)
    result = {"overall_success": overall, "n_tasks": n_tasks, "eps_per_task": eps_per_task,
              "per_task": per_task, "state_sanity": sane, "train_action_mse_note": "see libero BC"}
    with open(f"{VOL_PATH}/libero_rollout.json", "w") as f: json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({"overall_success": overall, "per_task": per_task}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def libero_smoke():
    """De-risk step: can we run the LIBERO sim headless on Modal at all? Import,
    build the libero_object suite, reset one env, render + step, report obs schema."""
    _bootstrap()
    import numpy as np
    import os
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bd = benchmark.get_benchmark_dict()
    print("benchmarks:", list(bd.keys()))
    suite = bd["libero_object"]()
    n = suite.n_tasks
    print(f"libero_object tasks: {n}")
    task = suite.get_task(0)
    print("task0:", task.language)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0); obs = env.reset()
    print("obs keys:", sorted(obs.keys()))
    for k in obs:
        v = np.asarray(obs[k])
        print(f"  {k}: {v.shape} {v.dtype}")
    a = np.zeros(env.action_dim if hasattr(env, "action_dim") else 7)
    for _ in range(3):
        obs, r, done, info = env.step(a)
    print(f"stepped OK; action_dim={len(a)}; agentview img shape="
          f"{np.asarray(obs.get('agentview_image')).shape}")
    env.close()
    return {"tasks": n, "task0": task.language, "obs_keys": sorted(obs.keys())}


def _bootstrap():
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)


@app.function(image=image, gpu="A100", timeout=300)
def a100_check():
    """Smoke test: is A100 unlocked now that a payment method is on file?
    Reports the GPU, does a trivial matmul, prints memory — a few seconds of A100."""
    _bootstrap()
    import torch
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    x = torch.randn(4096, 4096, device="cuda")
    y = (x @ x).sum().item()
    print(f"A100 OK ✅  device={name}  mem={total_gb:.1f}GB  matmul_ok={abs(y) > 0}")
    return {"device": name, "total_gb": round(total_gb, 1)}


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


def _cifar_loaders(batch=256, root="/vol/cifar"):
    import torch
    import torchvision as tv
    import torchvision.transforms as T
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tr = tv.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    te = tv.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, num_workers=4, drop_last=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False, num_workers=4))


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def train_vla_libero(steps: int = 4000, n_frames: int = 20000, horizon: int = 8, res: int = 64):
    H = horizon
    """Milestone 4b (HEADLINE): χ-VLA offline behavior-cloning on LIBERO-Object
    (real robot-imitation data; each task names a different object → language is
    load-bearing). Reports action-MSE + instruction-grounding gap + tensor-purity.
    Rollout success is DEFERRED (needs MuJoCo/robosuite)."""
    _bootstrap()
    import json
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig

    name = "lerobot/libero_object_image"
    # task_index → instruction string (lerobot meta; path varies by version).
    from huggingface_hub import list_repo_files
    tasks = {}
    files = list_repo_files(name, repo_type="dataset")
    task_files = [f for f in files if "task" in f.lower() and f.endswith((".jsonl", ".json", ".parquet"))]
    print(f"candidate task files: {task_files}")
    for tf in task_files:
        try:
            tp = hf_hub_download(name, tf, repo_type="dataset")
            if tp.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(tp).reset_index()
                print(f"  tasks.parquet columns: {list(df.columns)}")
                tcol = "task" if "task" in df.columns else \
                    next(c for c in df.columns if df[c].dtype == object)
                icol = "task_index" if "task_index" in df.columns else \
                    ("index" if "index" in df.columns else df.columns[0])
                for _, r in df.iterrows():
                    tasks[int(r[icol])] = str(r[tcol])
            else:
                for line in open(tp):
                    r = json.loads(line); tasks[int(r["task_index"])] = r["task"]
            if tasks:
                break
        except Exception as e:
            print(f"  {tf} failed: {e}")
    print(f"tasks ({len(tasks)}): {tasks}")

    # Build a word-level vocab from the instructions.
    words = set()
    for t in tasks.values():
        words.update(t.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words):
        vocab[w] = len(vocab)
    T = 32

    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        ids = ids[:T] + [0] * max(0, T - len(ids))
        return ids[:T]

    # Stream frames, resize, group into per-episode action chunks. Cache the raw
    # decoded frames to the volume so re-runs skip the slow HF streaming.
    import os, pickle
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    if os.path.exists(cache):
        print(f"loading cached frames from {cache}")
        with open(cache, "rb") as f:
            frames = pickle.load(f)
    else:
        print("streaming LIBERO frames...")
        ds = load_dataset(name, split="train", streaming=True)
        from PIL import Image
        frames = []
        for ex in ds:
            img = ex["observation.images.image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.resize((res, res))
            frames.append((int(ex["episode_index"]), int(ex["frame_index"]),
                           np.asarray(img, dtype=np.uint8),
                           np.asarray(ex["observation.state"], dtype=np.float32),
                           np.asarray(ex["action"], dtype=np.float32),
                           int(ex["task_index"])))
            if len(frames) >= n_frames:
                break
        with open(cache, "wb") as f:
            pickle.dump(frames, f)
        vol.commit()
    print(f"{len(frames)} frames loaded")

    # Group by episode; build (image_t, instr, state_t, action[t:t+H]) chunks.
    from collections import defaultdict
    eps = defaultdict(list)
    for f in frames:
        eps[f[0]].append(f)
    samples = []
    for ep, fs in eps.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            acts = np.stack([fs[i + k][4] for k in range(H)])   # (H, d_a)
            samples.append((fs[i][2], fs[i][5], fs[i][3], acts))
    print(f"{len(samples)} chunks; action dim {samples[0][3].shape[1]}")
    d_a = samples[0][3].shape[1]
    state_dim = samples[0][2].shape[0]

    # Normalize state + action by dataset stats (affine, foldable, spec §9/§11).
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    dev = "cuda"
    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    tidx = torch.tensor([s[1] for s in samples], device=dev)
    n = len(samples); ntr = int(n * 0.9)
    perm = torch.randperm(n, device=dev)
    tr, te = perm[:ntr], perm[ntr:]

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim,
                    n_embodiments=1, dim=384, n_layers=8, n_heads=12,
                    action_horizon=H, action_dim=d_a)
    model = ChiVLA(cfg).to(dev)
    print(f"χ-VLA params={model.num_params()/1e6:.2f}M  |samples|={n}")
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=8e-4, max_steps=steps, warmup_frac=0.05)
    def emb(bs):
        return torch.zeros(bs, dtype=torch.long, device=dev)

    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)
        idx = tr[torch.randint(len(tr), (256,), device=dev)]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs[idx], instr[idx], states[idx], emb(len(idx)), target_actions=actions[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 400 == 0:
            print(f"  step {step} mse {loss.item():.5f}")

    @torch.no_grad()
    def eval_mse(mode="full"):
        model.eval(); tot = cnt = 0.0
        for i in range(0, len(te), 512):
            b = te[i:i + 512]
            im, ins = imgs[b], instr[b]
            if mode == "shuffle_instr":
                ins = instr[b[torch.randperm(len(b), device=dev)]]
            elif mode == "state_only":            # zero image AND instruction
                im = torch.zeros_like(im); ins = torch.zeros_like(ins)
            elif mode == "zero_image":             # keep instruction+state, drop pixels
                im = torch.zeros_like(im)
            a, _ = model(im, ins, states[b], emb(len(b)))
            tot += (a - actions[b]).pow(2).sum().item(); cnt += a.numel()
        return tot / cnt

    mse = eval_mse("full")
    mse_sh = eval_mse("shuffle_instr")
    mse_state = eval_mse("state_only")
    mse_noimg = eval_mse("zero_image")
    # "mean predictor" is a trivial baseline: since actions are normalized to unit
    # variance, predicting the mean gives MSE ≈ 1.0 by construction. The honest
    # comparisons are the state-only model (robot trajectories are smooth and
    # state-predictable) and the input-ablation gaps.
    base = actions[te].var().item()
    result = {
        "n_samples": n, "n_tasks": len(tasks),
        "params_M": round(model.num_params() / 1e6, 2),
        "action_mse_full": round(mse, 5),
        "mse_state_only": round(mse_state, 5),      # zero image + instruction
        "mse_zero_image": round(mse_noimg, 5),      # keep instruction, zero pixels
        "mse_shuffled_instr": round(mse_sh, 5),
        "mean_predictor_mse": round(base, 5),       # trivial (≈1.0 by normalization)
        "gain_from_vision": round(mse_state - mse_noimg, 5),
        "gain_from_language": round(mse_sh - mse, 5),
        "state_only_already_explains": round(1 - mse_state / base, 3),
    }
    with open(f"{VOL_PATH}/vla_libero.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=2 * 3600)
def probe_libero(name: str = "lerobot/libero_object_image"):
    """De-risk M4b: load the LIBERO dataset and print its schema (keys, shapes)."""
    _bootstrap()
    import json
    from datasets import load_dataset
    try:
        ds = load_dataset(name, split="train", streaming=True)
        it = iter(ds)
        ex = next(it)
        info = {}
        for k, v in ex.items():
            try:
                import numpy as np
                arr = np.array(v)
                info[k] = {"type": type(v).__name__, "shape": list(arr.shape)[:4],
                           "dtype": str(arr.dtype)[:12]}
            except Exception:
                info[k] = {"type": type(v).__name__, "repr": str(v)[:80]}
        print("SCHEMA:", json.dumps(info, indent=2, default=str))
        return info
    except Exception as e:
        print(f"LOAD FAILED for {name}: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def odt_attention(layers: int = 1, steps: int = 1500):
    """M7 evidence: does global ODT structure survive through ATTENTION + RESIDUAL?
    Train a strict (foldable) χ-transformer LM via the distillation recipe
    (per-token teacher → near-identity scalar student, stable at depth), fold it
    (→ exact tensor network WITH attention), then at the block-INPUT bond — whose
    downstream crosses the attention layer(s) (a token feeds Q/K/V for every
    position) + residual + FFN + head — compare truncation by the GLOBAL
    output-sensitivity Gram vs a LOCAL adjacent-weight SVD. Global > local ⇒ ODT
    low-rank structure is real through attention, not just in feedforward chains."""
    _bootstrap()
    import json
    from dataclasses import replace
    import torch
    from xvla.models.lm import ChiLanguageModel, LMConfig
    from xvla.nn.normalization import RmsBatchNorm
    from xvla.train.data import prepare_tinyshakespeare, TokenBinDataset
    from xvla.train.train_lm import _lr_at, TrainConfig
    from xvla.train.calibrate import calibrate_rbn
    from xvla.train.distill import DistillConfig, distill_strict
    from xvla.train.fold import verify_fold

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()
    dev = "cuda"
    torch.manual_seed(0)
    ds = TokenBinDataset(train_bin, 128); vds = TokenBinDataset(val_bin, 128)

    # 1) Train a stable per-token teacher.
    tcfg = LMConfig(vocab_size=vocab, dim=128, n_layers=layers, n_heads=4, max_seq_len=128,
                    attn="bilinear", ffn="bilinear", norm="per_token", qk_norm="per_token")
    tc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=128, batch_size=32,
                     lr=6e-4, max_steps=steps, warmup_frac=0.05)
    teacher = ChiLanguageModel(tcfg).to(dev)
    opt = torch.optim.AdamW(teacher.parameters(), lr=tc.lr, betas=(0.9, 0.95), weight_decay=0.1)
    teacher.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tc)
        x, y = ds.batch(32, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = teacher(x, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"  [teacher] step {step} loss {loss.item():.4f}")

    # 2) Distill into a strict scalar-norm student with near-identity gains.
    scfg = replace(tcfg, norm="scalar_rbn", qk_norm="scalar_rbn", learned_gain=True)
    dtc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=128, batch_size=32,
                      lr=1e-4, weight_decay=0.5, grad_clip=0.3, balance_every=25,
                      max_steps=steps, warmup_frac=0.1, eval_every=steps, log_every=500)
    dh = distill_strict(teacher, scfg, dtc, DistillConfig(lambda_feat=0.0, lambda_ce=0.5))
    model = dh["student"]
    print(f"  student distilled: strict val {dh['final_val_loss']:.4f}")

    calibrate_rbn(model, lambda: ds.batch(32, dev), iters=80)
    for m in model.modules():
        if isinstance(m, RmsBatchNorm):
            m.frozen = True
    model = model.float().eval()

    # (1) fold → exact tensor network WITH attention.
    xf, _ = vds.batch(8, dev)
    _, abs32, rel32 = verify_fold(model, xf)
    _, abs64, _ = verify_fold(model.double(), xf.long())
    model = model.float()
    print(f"FOLD (attention model): rel {rel32:.2e} fp32 / abs {abs64:.2e} fp64")

    # Forward from a projected block input (bond = token+pos embeddings).
    def forward_proj(idx, targets, P=None):
        pos = torch.arange(idx.shape[1], device=dev)
        x0 = model.tok_emb(idx) + model.pos_emb(pos)[None]
        if P is not None:
            x0 = x0 @ P.T
        h = x0
        for blk in model.blocks:
            h = blk(h, method="explicit")
        h = model.rbn_out(h)
        logits = model.head(h)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    # (2) Global Gram = output-sensitivity at the block-input bond (accounts for
    # attention mixing + residual + FFN + head). Local = adjacent input-weight SVD.
    d = model.cfg.dim
    G_global = torch.zeros(d, d, device=dev)
    G_actcov = torch.zeros(d, d, device=dev)
    nb = 0
    for _ in range(30):
        idx, y = vds.batch(32, dev)
        pos = torch.arange(idx.shape[1], device=dev)
        x0 = (model.tok_emb(idx) + model.pos_emb(pos)[None]).detach().requires_grad_(True)
        h = x0
        for blk in model.blocks:
            h = blk(h, method="explicit")
        loss = torch.nn.functional.cross_entropy(
            model.head(model.rbn_out(h)).reshape(-1, vocab), y.reshape(-1))
        g, = torch.autograd.grad(loss, x0)
        gf = g.reshape(-1, d); xf2 = x0.detach().reshape(-1, d)
        G_global += gf.T @ gf; G_actcov += xf2.T @ xf2; nb += gf.shape[0]
    G_global /= nb; G_actcov /= nb
    # local: SVD of the stacked adjacent input projections (Q1,K1,Q2,K2,V,FFN L,R)
    b0 = model.blocks[0]
    W = torch.cat([b0.attn.wq1.weight, b0.attn.wk1.weight, b0.attn.wq2.weight,
                   b0.attn.wk2.weight, b0.attn.wv.weight,
                   b0.ffn.left.weight, b0.ffn.right.weight], dim=0)  # (*, d)
    G_local = W.T @ W

    def topk_proj(G, k):
        ev, V = torch.linalg.eigh(G); V = V.flip(1)
        return V[:, :k] @ V[:, :k].T

    val_batches = [vds.batch(32, dev) for _ in range(20)]
    def val_loss(P):
        with torch.no_grad():
            return sum(forward_proj(i, y, P).item() for i, y in val_batches) / len(val_batches)
    full = val_loss(None)
    ks = [4, 8, 16, 32, 64, 96, 128]
    curves = {"global": {}, "local_wsvd": {}, "act_pca": {}}
    for k in ks:
        curves["global"][k] = round(val_loss(topk_proj(G_global, k)), 4)
        curves["local_wsvd"][k] = round(val_loss(topk_proj(G_local, k)), 4)
        curves["act_pca"][k] = round(val_loss(topk_proj(G_actcov, k)), 4)
    removable = max([d - k for k in ks if curves["global"][k] <= full + 0.05] + [0])

    print(f"full val loss {full:.4f}")
    print("  k  | global | local_wsvd | act_pca")
    for k in ks:
        print(f"  {k:3d} | {curves['global'][k]:.3f}  | {curves['local_wsvd'][k]:.3f}     | {curves['act_pca'][k]:.3f}")
    result = {
        "layers": layers, "dim": d, "full_val_loss": round(full, 4),
        "fold_rel_fp32": rel32, "fold_abs_fp64": abs64,
        "curves": curves, "removable_dims_0.05nats": removable,
        "global_beats_local_at": [k for k in ks if curves["global"][k] < curves["local_wsvd"][k] - 1e-3],
    }
    with open(f"{VOL_PATH}/odt_attention.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=2))
    return result


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=300)
def read_result(fname: str):
    """Print a JSON result file from the volume (for polling detached runs)."""
    import json, os
    p = f"{VOL_PATH}/{fname}"
    if not os.path.exists(p):
        print("NOT_READY"); return None
    print("RESULT:", open(p).read())
    return json.load(open(p))


@app.function(image=image, gpu="A10G", timeout=1800)
def attention_latency_crossover():
    """M6/M7 deliverable: measure explicit vs Khatri-Rao bilinear-attention latency
    across sequence length N (spec §6.5: KR wins when N > d_h²). A real, defensible
    kernel-latency artifact for the scale story (no A100 needed)."""
    _bootstrap()
    import json, time
    import torch
    from xvla.nn.attention import BilinearAttention

    dev = "cuda"
    dim, heads = 512, 16          # d_h = 32 → d_h² = 1024 crossover prediction
    attn = BilinearAttention(dim, heads, causal=True, qk_norm="per_token").to(dev).eval()
    dh2 = (dim // heads) ** 2
    rows = []
    for N in [128, 256, 512, 1024, 2048, 4096]:
        x = torch.randn(2, N, dim, device=dev)
        def bench(method):
            for _ in range(3):
                with torch.no_grad(): attn(x, method=method)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10):
                with torch.no_grad(): attn(x, method=method)
            torch.cuda.synchronize()
            return (time.time() - t) / 10 * 1e3   # ms
        try:
            e = bench("explicit")
        except RuntimeError:
            e = float("inf")
        try:
            k = bench("khatri_rao")
        except RuntimeError:
            k = float("inf")
        rows.append({"N": N, "explicit_ms": round(e, 3), "khatri_rao_ms": round(k, 3),
                     "kr_faster": k < e})
        print(f"N={N:5d}  explicit {e:.3f}ms  khatri_rao {k:.3f}ms  KR_faster={k<e}")
    result = {"d_h2_crossover_pred": dh2, "rows": rows}
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", timeout=2 * 3600)
def train_vla_synth(steps: int = 3000, dual: bool = False, projector: str = "crossbilinear"):
    """Milestone 4a: train χ-VLA on the synthetic reach task; report action-MSE,
    language-blind lower bound, and the instruction-grounding gap (shuffled-instr
    MSE − correct-instr MSE > 0 ⇒ language is used)."""
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.synth_vla import make_batch, VOCAB
    from xvla.train.train_lm import _lr_at, TrainConfig

    torch.manual_seed(0)
    dev = "cuda"
    # Pre-generate pools once (make_batch has a per-sample python loop).
    print("generating data pools...")
    train_pool = make_batch(8192, dev)
    test_pool = make_batch(2048, dev)
    test_shuf = make_batch(2048, dev, shuffle_instr=True)

    cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                    dual_vision=dual, projector=projector, vocab_size=VOCAB, max_instr_len=16,
                    state_dim=8, n_embodiments=4, dim=256, n_layers=6, n_heads=8,
                    action_horizon=4, action_dim=7)
    model = ChiVLA(cfg).to(dev)
    print(f"χ-VLA params={model.num_params()/1e6:.2f}M  dual={dual} proj={projector}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    tccfg = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)

    def sample(pool, bs=256):
        idx = torch.randint(pool["img"].shape[0], (bs,), device=dev)
        return (pool["img"][idx], pool["instr"][idx], pool["state"][idx],
                pool["embodiment"][idx], pool["actions"][idx])

    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tccfg)
        img, instr, state, emb, act = sample(train_pool)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(img, instr, state, emb, target_actions=act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"  step {step} mse {loss.item():.5f}")

    @torch.no_grad()
    def eval_mse(pool):
        model.eval()
        tot = totp = n = 0.0
        for i in range(0, pool["img"].shape[0], 512):
            sl = slice(i, i + 512)
            a, _ = model(pool["img"][sl], pool["instr"][sl], pool["state"][sl],
                         pool["embodiment"][sl])
            tot += (a - pool["actions"][sl]).pow(2).sum().item(); n += pool["actions"][sl].numel()
            totp += (a[..., :2] - pool["actions"][sl][..., :2]).pow(2).sum().item()
        model.train()
        return tot / n, totp / (n * 2 / 7)      # full mse, position-only mse

    # position-only language-blind lower bound
    bp = ((test_pool["actions"][..., :2].mean(0, keepdim=True) - test_pool["actions"][..., :2])
          .pow(2).mean().item())  # best constant predictor on position
    mse, mse_pos = eval_mse(test_pool)
    mse_shuf, mse_shuf_pos = eval_mse(test_shuf)
    result = {
        "dual": dual, "projector": projector if dual else "single",
        "params_M": round(model.num_params() / 1e6, 2),
        "action_mse_full": round(mse, 6),
        "action_mse_position": round(mse_pos, 6),
        "blind_position_mse": round(bp, 6),
        "shuffled_instr_position_mse": round(mse_shuf_pos, 6),
        "grounding_gap_position": round(mse_shuf_pos - mse_pos, 6),
        "beats_blind_position": mse_pos < bp,
    }
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", timeout=1800)
def m3_projector(steps: int = 4000, ncls: int = 10):
    """M3: cross-bilinear CP projector vs concat baselines on a target that is
    BILINEAR BY CONSTRUCTION (label = argmax of a fixed spatial×semantic form,
    no linear part) — linearly inseparable, so only a model with the interaction
    term can fit it. Proves the cross-bilinear interaction does real work."""
    _bootstrap()
    import json
    import torch
    import torch.nn.functional as F
    from xvla.nn.projector import CrossBilinearProjector, ConcatLinear

    torch.manual_seed(0)
    dev = "cuda"
    ds = dm = 32
    h = 16
    # Fixed bilinear TEACHER: y = argmax_c D*[ (Ls* s) ⊙ (Rm* m) ], purely a
    # spatial×semantic product — no additive/linear term the baselines could use.
    Ls_t = torch.randn(h, ds, device=dev)
    Rm_t = torch.randn(h, dm, device=dev)
    D_t = torch.randn(ncls, h, device=dev)

    def batch(bs=512):
        s = torch.randn(bs, ds, device=dev)
        m = torch.randn(bs, dm, device=dev)
        with torch.no_grad():
            y = (D_t @ ((s @ Ls_t.T) * (m @ Rm_t.T)).T).T.argmax(-1)
        return s, m, y

    def train(model, interaction=True):
        model = model.to(dev)
        if hasattr(model, "use_interaction"):
            model.use_interaction = interaction
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        model.train()
        for _ in range(steps):
            s, m, y = batch()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(s, m), y)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            cor = tot = 0
            for _ in range(20):
                s, m, y = batch(1000)
                cor += (model(s, m).argmax(-1) == y).sum().item(); tot += y.numel()
        return cor / tot, sum(p.numel() for p in model.parameters())

    xbil = CrossBilinearProjector(ds, dm, ncls, rank=2 * ncls)
    acc_bil, p_bil = train(xbil, interaction=True)
    xabl = CrossBilinearProjector(ds, dm, ncls, rank=2 * ncls)
    acc_abl, _ = train(xabl, interaction=False)          # interaction zeroed
    acc_lin, p_lin = train(ConcatLinear(ds, dm, ncls, depth=1))
    # param-match the MLP hidden to the cross-bilinear param count
    hidden = max(8, (p_bil - ncls) // (ds + dm + ncls))
    acc_mlp, p_mlp = train(ConcatLinear(ds, dm, ncls, depth=2, hidden=hidden))

    result = {
        "n_classes": ncls, "chance": round(1 / ncls, 3),
        "cross_bilinear": {"acc": round(acc_bil, 4), "params": p_bil},
        "cross_bilinear_no_interaction": {"acc": round(acc_abl, 4)},
        "concat_linear": {"acc": round(acc_lin, 4), "params": p_lin},
        "concat_mlp_gelu": {"acc": round(acc_mlp, 4), "params": p_mlp, "note": "non-pure baseline"},
        "gap_bilinear_minus_concat_linear": round(acc_bil - acc_lin, 4),
        "interaction_ablation_drop": round(acc_bil - acc_abl, 4),
    }
    print("RESULT:", json.dumps(result, indent=2))
    return result


def _svhn_loaders(batch=256, root="/vol/svhn", flatten=False):
    import torch
    import torchvision as tv
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])
    tr = tv.datasets.SVHN(root, split="train", download=True, transform=tf)
    te = tv.datasets.SVHN(root, split="test", download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, num_workers=4, drop_last=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False, num_workers=4))


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def odt_experiment(epochs: int = 20):
    """M5 FLAGSHIP: train a feedforward χ-MLP on SVHN, then exact ODT —
    (i) exact reconstruction from cores, (ii) global low-rank (dims removable),
    (iii) global ODT vs local SVD at matched rank on the deepest bond."""
    _bootstrap()
    import json, math
    import torch
    from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
    from xvla.train.odt import (export_cores, unroll_forward, downstream_gram,
                                local_gram, truncation_curve, top_projector, accuracy)

    tl, vl = _svhn_loaders(flatten=True)
    vol.commit()
    torch.manual_seed(0)
    cfg = ChiMLPConfig(in_dim=3072, dim=32, n_layers=3, num_classes=10, norm="scalar_rbn")
    model = ChiMLP(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05, betas=(0.9, 0.95))
    print(f"χ-MLP params={model.num_params()/1e6:.2f}M")
    model.train()
    for ep in range(epochs):
        for imgs, labels in tl:
            imgs, labels = imgs.cuda(), labels.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(imgs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # quick eval
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for imgs, labels in vl:
                lg, _ = model(imgs.cuda())
                cor += (lg.argmax(-1).cpu() == labels).sum().item(); tot += labels.numel()
        model.train()
        print(f"  epoch {ep} acc {cor/tot:.4f}")
    base_acc = cor / tot

    # Gather a test tensor for exact analysis (fp64).
    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 4000:
            break
    x = torch.cat(xs)[:4000].cuda(); y = torch.cat(ys)[:4000]

    embed, cores, head = export_cores(model)
    embed = tuple(t.cuda() for t in embed); cores = [c.cuda() for c in cores]
    head = tuple(t.cuda() for t in head)

    # (i) exact reconstruction: unrolled cores vs module forward.
    model.eval()
    with torch.no_grad():
        logits_mod = model.double()(x.double())[0]
    logits_unroll = unroll_forward(embed, cores, head, x)
    recon = (logits_mod - logits_unroll).abs().max().item()
    acc_unroll = accuracy(logits_unroll, y.cuda())
    print(f"(i) reconstruction max|Δ| = {recon:.2e}  | unroll acc {acc_unroll:.4f}")

    # (ii)+(iii) deepest bond (bond=1: a_1 feeds layers 2,3).
    bond = 1
    G_glob = downstream_gram(cores, head, bond)
    G_loc = local_gram(cores, bond)
    ks = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    cg = truncation_curve(embed, cores, head, bond, G_glob, x, y.cuda(), ks)
    cl = truncation_curve(embed, cores, head, bond, G_loc, x, y.cuda(), ks)
    _, spec = top_projector(G_glob, 32)
    spec = spec.clamp_min(0)
    # dims removable at <=1% acc drop (global)
    full = cg[32]
    removable = max([32 - k for k in ks if cg[k] >= full - 0.01] + [0])

    print("(iii) rank |  global ODT | local SVD")
    for k in ks:
        print(f"      {k:3d}  |   {cg[k]:.4f}    | {cl[k]:.4f}")
    print(f"(ii) full acc {full:.4f}; removable dims @≤1% drop = {removable}/32 "
          f"({100*removable/32:.0f}%)")

    result = {
        "base_acc": base_acc, "unroll_acc": acc_unroll,
        "reconstruction_max_abs": recon,
        "global_curve": cg, "local_curve": cl,
        "removable_dims_1pct": removable, "removable_frac": removable / 32,
        "spectrum": [round(float(s), 6) for s in spec.tolist()],
        "global_beats_local_at": [k for k in ks if cg[k] > cl[k] + 1e-4],
    }
    with open(f"{VOL_PATH}/odt_experiment.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "spectrum"}, indent=2))
    return result


def _train_chi_mlp(dim, n_layers, epochs, tl, vl):
    """Train a χ-MLP on SVHN and return (model, val_acc). Shared by ODT experiments."""
    import torch
    from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
    torch.manual_seed(0)
    cfg = ChiMLPConfig(in_dim=3072, dim=dim, n_layers=n_layers, num_classes=10, norm="scalar_rbn")
    model = ChiMLP(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05, betas=(0.9, 0.95))
    model.train()
    acc = 0.0
    for ep in range(epochs):
        for imgs, labels in tl:
            imgs, labels = imgs.cuda(), labels.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(imgs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for imgs, labels in vl:
                lg, _ = model(imgs.cuda())
                cor += (lg.argmax(-1).cpu() == labels).sum().item(); tot += labels.numel()
        model.train(); acc = cor / tot
    model.eval()   # freeze RmsBatchNorm to its running EMA so the fold is exact
    print(f"  χ-MLP dim={dim} L={n_layers}: val acc {acc:.4f}")
    return model, acc


def _save_atom_grid(rows, path, cell=56, pad=3, row_labels=None, split_after=None,
                    gamma=0.7, margin=22):
    """Save a labeled montage PNG. ``rows`` is a list of lists of (3,32,32) fp
    tensors; each atom is a grayscale magnitude map (bright = high sensitivity),
    contrast-stretched per atom and gamma-brightened. ``row_labels`` prints a tag
    left of each row; ``split_after`` inserts a divider column (e.g. between
    supporting and suppressing atoms). Diverging sign is in JSON, not colour."""
    import numpy as np
    from PIL import Image, ImageDraw
    ncol = max(len(r) for r in rows)
    gap = pad + (6 if split_after is not None else 0)
    H = len(rows) * (cell + pad) + pad + margin
    W = margin + ncol * (cell + pad) + pad + (gap if split_after is not None else 0)
    canvas = np.full((H, W), 22, dtype=np.uint8)
    for ri, row in enumerate(rows):
        for ci, atom in enumerate(row):
            mag = (atom ** 2).sum(0).sqrt().cpu().numpy()      # (32,32) magnitude
            mag = mag - mag.min()
            mag = (mag / (mag.max() + 1e-12)) ** gamma          # brighten mid-tones
            img = Image.fromarray((mag * 255).astype(np.uint8)).resize(
                (cell, cell), Image.NEAREST)
            xshift = gap if (split_after is not None and ci >= split_after) else 0
            y0 = margin + pad + ri * (cell + pad)
            x0 = margin + pad + ci * (cell + pad) + xshift
            canvas[y0:y0 + cell, x0:x0 + cell] = np.asarray(img)
    im = Image.fromarray(canvas).convert("L")
    draw = ImageDraw.Draw(im)
    if row_labels:
        for ri, lab in enumerate(row_labels):
            draw.text((4, margin + pad + ri * (cell + pad) + cell // 2 - 4),
                      str(lab), fill=255)
    if split_after is not None:
        draw.text((margin + pad + 2, 6), "supporting  (lambda>0)", fill=200)
        xg = margin + pad + split_after * (cell + pad) + gap - 4
        draw.text((xg + 2, 6), "suppressing  (lambda<0)", fill=200)
    im.save(path)


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def odt_interpret(epochs: int = 15):
    """ACTUALLY apply ODT + test whether we can interpret (spec §16.2-A / §17).

    Extends M5 beyond truncation numbers, and extends Dehérand (2026) from
    shallow-conv to (A) an exact output-conditioned eigenspectrum with a *causal*
    faithfulness battery, and (B) the deep multi-layer case he did not attempt.

    Part A — shallow single-bilinear χ-classifier: logit_c = z^T Q_c z + b_c
      (i)   exact reconstruction of the quadratic form vs the module;
      (ii)  eigen-atoms of each Q_c → pixel visual atoms (PNG grid);
      (iii) coherence: atom spatial-locality vs random-direction control;
      (iv)  faithfulness: keep-top-r / drop-top-r / drop-random-r accuracy +
            an amplify-the-atom directed test.
    Part B — deep 3-layer χ-MLP: global visual atoms at the deepest bond, and
      global-vs-random subspace truncation (the causal control for the ranking).
    """
    _bootstrap()
    import json
    import torch
    from xvla.train.odt import (export_cores, unroll_forward, downstream_gram,
                                top_projector, random_projector, accuracy)
    from xvla.train.odt_interp import (class_quadratics, eig_atoms, quad_logits,
                                       atom_to_pixels, locality, faithfulness,
                                       amplify_test)

    tl, vl = _svhn_loaders(flatten=True)
    vol.commit()
    IN_SHAPE = (3, 32, 32)
    std = torch.tensor([0.198, 0.201, 0.197]).view(3, 1, 1).repeat(1, 32, 32).cuda()

    # A test tensor for exact fp64 analysis.
    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 4000:
            break
    x = torch.cat(xs)[:4000].cuda(); y = torch.cat(ys)[:4000].cuda()
    rng = torch.Generator(device="cuda").manual_seed(0)

    # ===================== Part A: shallow, output-conditioned ================ #
    model_s, acc_s = _train_chi_mlp(dim=64, n_layers=1, epochs=epochs, tl=tl, vl=vl)
    embed, cores, head = export_cores(model_s)
    embed = tuple(t.cuda() for t in embed); cores = [c.cuda() for c in cores]
    head = tuple(t.cuda() for t in head)

    # (i) exact reconstruction of the QUADRATIC form (not just the unroll).
    Q, b = class_quadratics(cores, head)
    with torch.no_grad():
        logits_mod = model_s.double()(x.double())[0]
    logits_quad = quad_logits(Q, b, x, embed)
    recon = (logits_mod - logits_quad).abs().max().item()
    acc_quad = accuracy(logits_quad, y)
    print(f"A(i) quadratic-form reconstruction max|Δ| = {recon:.2e} | acc {acc_quad:.4f}")

    # (ii) eigen-atoms → pixel visual atoms; (iii) coherence vs random control.
    n_pos, n_neg = 3, 3
    atom_rows, atom_meta = [], []
    rand_loc = []
    for _ in range(200):                                  # random-direction locality baseline
        v = torch.randn(Q.shape[1], device="cuda", dtype=torch.float64)
        rand_loc.append(locality(atom_to_pixels(v, embed, IN_SHAPE, std)))
    rand_loc = float(sum(rand_loc) / len(rand_loc))
    for c in range(10):
        l, v = eig_atoms(Q[c])
        pos_idx = (l > 0).nonzero().flatten()[:n_pos]
        neg_idx = (l < 0).nonzero().flatten()
        neg_idx = neg_idx[l[neg_idx].abs().argsort(descending=True)][:n_neg]
        row, meta = [], []
        for idx, sign in [(pos_idx, "+"), (neg_idx, "-")]:
            for i in idx:
                p = atom_to_pixels(v[:, i], embed, IN_SHAPE, std)
                row.append(p)
                meta.append({"class": c, "sign": sign,
                             "eig": round(float(l[i]), 4),
                             "locality": round(locality(p), 4)})
        atom_rows.append(row); atom_meta.extend(meta)
    _save_atom_grid(atom_rows, f"{VOL_PATH}/odt_atoms_shallow.png",
                    row_labels=list(range(10)), split_after=n_pos)
    atom_loc = sum(m["locality"] for m in atom_meta) / len(atom_meta)
    print(f"A(iii) mean atom locality {atom_loc:.3f} vs random {rand_loc:.3f} "
          f"(ratio {atom_loc/rand_loc:.2f}×)")

    # (iv) faithfulness battery + amplify test.
    ranks = [1, 2, 4, 8, 16, 32]
    faith = faithfulness(Q, b, x, y, embed, ranks, rng=rng)
    amps = [amplify_test(Q, b, x, y, embed, c, scale=3.0) for c in range(10)]
    print("A(iv) rank | keep  drop_top  drop_random")
    for r in ranks:
        print(f"       {r:3d} | {faith['keep'][r]:.3f}  {faith['drop_top'][r]:.3f}"
              f"     {faith['drop_random'][r]:.3f}")

    # ===================== Part B: deep global atoms ========================== #
    model_d, acc_d = _train_chi_mlp(dim=32, n_layers=3, epochs=epochs, tl=tl, vl=vl)
    embed_d, cores_d, head_d = export_cores(model_d)
    embed_d = tuple(t.cuda() for t in embed_d); cores_d = [c.cuda() for c in cores_d]
    head_d = tuple(t.cuda() for t in head_d)
    bond = 1
    G = downstream_gram(cores_d, head_d, bond)
    # Global directions of a_bond → each is a quadratic form in z0 → pixel atom.
    _, gvecs = torch.linalg.eigh(G)
    gvecs = gvecs.flip(1)                                  # top global directions first
    C1 = cores_d[0]                                        # (d, d+1, d+1): a_1 = C1(z0,z0)
    deep_rows, deep_meta = [], []
    for j in range(6):                                     # top-6 global a_1 directions
        # a_1·v = z0^T (Σ_o v_o C1[o]) z0 : the global direction as a quadratic in z0.
        M = torch.einsum("o,oij->ij", gvecs[:, j].double(), C1)
        M = 0.5 * (M + M.T)
        lm, vm = eig_atoms(M)
        p = atom_to_pixels(vm[:, 0], embed_d, IN_SHAPE, std)
        deep_rows.append(p)
        deep_meta.append({"global_dir": j, "locality": round(locality(p), 4)})
    _save_atom_grid([deep_rows], f"{VOL_PATH}/odt_atoms_deep.png",
                    row_labels=["a1"])
    deep_loc = sum(m["locality"] for m in deep_meta) / len(deep_meta)

    # Global vs random subspace truncation at the bond (causal control).
    ks = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    glob_curve, rand_curve = {}, {}
    for k in ks:
        Pg, _ = top_projector(G, k)
        glob_curve[k] = accuracy(unroll_forward(embed_d, cores_d, head_d, x,
                                                proj=Pg, proj_bond=bond), y)
        Pr = random_projector(G.shape[0], k, device="cuda", rng=rng)
        rand_curve[k] = accuracy(unroll_forward(embed_d, cores_d, head_d, x,
                                                proj=Pr, proj_bond=bond), y)
    print("B rank | global  random")
    for k in ks:
        print(f"    {k:3d} | {glob_curve[k]:.3f}  {rand_curve[k]:.3f}")

    result = {
        "shallow": {
            "val_acc": acc_s, "quad_recon_max_abs": recon, "quad_acc": acc_quad,
            "atom_locality": round(atom_loc, 4), "random_locality": round(rand_loc, 4),
            "locality_ratio": round(atom_loc / rand_loc, 3),
            "faithfulness": faith,
            "amplify": amps,
            "atoms": atom_meta,
        },
        "deep": {
            "val_acc": acc_d, "atom_locality": round(deep_loc, 4),
            "global_curve": glob_curve, "random_curve": rand_curve,
            "global_beats_random_at": [k for k in ks
                                       if glob_curve[k] > rand_curve[k] + 1e-4],
            "atoms": deep_meta,
        },
    }
    with open(f"{VOL_PATH}/odt_interpret.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({
        "shallow_acc": acc_s, "quad_recon": recon, "locality_ratio": result["shallow"]["locality_ratio"],
        "keep@4": faith["keep"][4], "drop_top@4": faith["drop_top"][4],
        "drop_random@4": faith["drop_random"][4],
        "deep_acc": acc_d, "deep_global_beats_random_at": result["deep"]["global_beats_random_at"],
    }, indent=2))
    return result


def _train_shallow(model, tl, vl, epochs, lr=2e-3):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    model.cuda().train()
    acc = 0.0
    for ep in range(epochs):
        for imgs, labels in tl:
            imgs, labels = imgs.cuda(), labels.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(imgs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for imgs, labels in vl:
                lg, _ = model(imgs.cuda())
                cor += (lg.argmax(-1).cpu() == labels).sum().item(); tot += labels.numel()
        model.train(); acc = cor / tot
    model.eval()
    return acc


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def interp_baselines_clf(epochs: int = 12, seeds: int = 2):
    """G1b/G7: the DATA-FREE claim, done right. On the exact feedforward χ-classifier
    (conv-spatial, SVHN), rank input directions by (a) ODT [DATA-FREE, from weights:
    eigvecs of Σ_c Q_c²], (b) PCA of inputs [data-driven], (c) integrated gradients
    [data-driven, standard attribution], (d) random; score by deletion/insertion AUC.
    If data-free ODT ≈ data-heavy IG/PCA ≫ random ⇒ 'faithful attribution for free from
    weights' — the defensible version of claim 1."""
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_conv import ShallowBilinear
    from xvla.train.topology import class_quadratics

    tl, vl = _svhn_loaders(flatten=False)
    vol.commit()
    in_ch, hw = 3, 32; D = in_ch * hw * hw
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3072]
    agg = {m: {"deletion_auc": [], "insertion_auc": []} for m in ["odt", "pca", "ig", "random"]}
    accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = ShallowBilinear(mode="conv", readout="spatial", grid=8, width=48, kernel=5,
                                in_ch=in_ch, hw=hw, num_classes=10)
        acc = _train_shallow(model, tl, vl, epochs, lr=2e-3); accs.append(acc)
        model = model.cuda().double().eval()
        xs, ys = [], []
        for imgs, labels in vl:
            xs.append(imgs); ys.append(labels)
            if sum(t.shape[0] for t in xs) >= 3000:
                break
        X = torch.cat(xs)[:3000].cuda().double(); Y = torch.cat(ys)[:3000].cuda()

        # (a) ODT data-free importance Gram Σ_c Q_c²
        Q, _, _ = class_quadratics(model, "cuda")
        G_odt = torch.zeros(D, D, device="cuda", dtype=torch.float64)
        for c in range(Q.shape[0]):
            G_odt += Q[c] @ Q[c]
        V_odt = torch.linalg.eigh(G_odt)[1].flip(1); del Q, G_odt; torch.cuda.empty_cache()
        # (b) PCA of inputs
        Xf = X.reshape(-1, D); Xm = Xf.mean(0)
        V_pca = torch.linalg.eigh((Xf - Xm).T @ (Xf - Xm) / Xf.shape[0])[1].flip(1)
        # (c) integrated gradients importance Gram
        attrs = torch.zeros(Xf.shape[0], D, device="cuda", dtype=torch.float64)
        st = 20
        for i in range(0, Xf.shape[0], 250):
            x = X[i:i+250]; y = Y[i:i+250]; tot = torch.zeros_like(x)
            for s in range(1, st + 1):
                xs_ = ((s / st) * x).detach().requires_grad_(True)
                with torch.enable_grad():
                    sel = model.logits(xs_).gather(1, y[:, None]).sum()
                    gg, = torch.autograd.grad(sel, xs_)
                tot = tot + gg
            attrs[i:i+250] = (x * tot / st).reshape(x.shape[0], -1)
        V_ig = torch.linalg.eigh(attrs.T @ attrs / attrs.shape[0])[1].flip(1)
        del attrs; torch.cuda.empty_cache()
        rng = torch.Generator(device="cuda").manual_seed(seed)
        V_rand = torch.linalg.qr(torch.randn(D, D, generator=rng, device="cuda", dtype=torch.float64))[0]

        @torch.no_grad()
        def acc_proj(V, k, keep):
            P = V[:, :k] @ V[:, :k].T
            if not keep:
                P = torch.eye(D, device="cuda", dtype=torch.float64) - P
            c = t = 0
            for i in range(0, Xf.shape[0], 500):
                xp = (Xf[i:i+500] @ P).reshape(-1, in_ch, hw, hw)
                lg = model.logits(xp)
                c += (lg.argmax(-1) == Y[i:i+500]).sum().item(); t += lg.shape[0]
            return c / t
        fr = [k / D for k in ks]
        def auc(v): return sum((fr[i+1]-fr[i])*(v[i]+v[i+1])/2 for i in range(len(fr)-1)) / (fr[-1]-fr[0])
        for name, V in [("odt", V_odt), ("pca", V_pca), ("ig", V_ig), ("random", V_rand)]:
            agg[name]["deletion_auc"].append(auc([acc_proj(V, k, False) for k in ks]))
            agg[name]["insertion_auc"].append(auc([acc_proj(V, k, True) for k in ks]))
        torch.cuda.empty_cache()

    def ms(v):
        t = torch.tensor(v); return [round(t.mean().item(), 3), round(t.std().item(), 3)]
    result = {"clf_acc": ms(accs), **{m: {k: ms(v) for k, v in d.items()} for m, d in agg.items()},
              "note": "deletion↓ insertion↑; odt is DATA-FREE (weights only), pca+ig are data-driven"}
    with open(f"{VOL_PATH}/interp_baselines_clf.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def vla_bond_seeds(steps: int = 3000, seeds: int = 3):
    """G5: error bars on the bond-decodability table (C-v2). Linear-probe R² for target
    position at the input bond vs the post-attention action-query bond, over multiple seeds."""
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.synth_vla import make_batch, VOCAB
    from xvla.train.train_lm import _lr_at, TrainConfig
    dev = "cuda"; H = 4
    inp_r2, post_r2 = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        tp = make_batch(8192, dev); probe = make_batch(3000, dev)
        cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                        vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=4,
                        dim=256, n_layers=6, n_heads=8, action_horizon=4, action_dim=7)
        m = ChiVLA(cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        tc = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)
        m.train()
        for step in range(steps + 1):
            for g in opt.param_groups: g["lr"] = _lr_at(step, tc)
            i = torch.randint(8192, (256,), device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = m(tp["img"][i], tp["instr"][i], tp["state"][i], tp["embodiment"][i],
                            target_actions=tp["actions"][i])
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        @torch.no_grad()
        def feats(img, instr, state, emb):
            B = img.shape[0]
            x = torch.cat([m._visual_tokens(img), m.bos.expand(B, -1, -1), m.tok_emb(instr),
                           m.state_proj(state)[:, None], m.embodiment_emb(emb)[:, None],
                           m.action_queries.expand(B, -1, -1)], dim=1)
            x = x + m.pos_emb[:, :x.shape[1]]
            inp = x.mean(1)
            xo = m.norm_out(m.backbone(x, mask=causal_mask(x.shape[1], device=dev, dtype=x.dtype)))
            return inp.double(), xo[:, -H:].mean(1).double()
        A = [feats(probe["img"][i:i+512], probe["instr"][i:i+512], probe["state"][i:i+512],
                   probe["embodiment"][i:i+512]) for i in range(0, probe["img"].shape[0], 512)]
        inp = torch.cat([a[0] for a in A]); post = torch.cat([a[1] for a in A])
        tgt = torch.stack([probe["state"][:, 0] + H * probe["actions"][:, 0, 0],
                           probe["state"][:, 1] + H * probe["actions"][:, 0, 1]], 1).double()
        N = inp.shape[0]; ntr = int(0.7 * N)
        def r2(F):
            Fa = torch.cat([F[:ntr], torch.ones(ntr, 1, device=dev, dtype=torch.float64)], 1)
            W = torch.linalg.solve(Fa.T @ Fa + 1e-2 * torch.eye(Fa.shape[1], device=dev, dtype=torch.float64), Fa.T @ tgt[:ntr])
            Fte = torch.cat([F[ntr:], torch.ones(N - ntr, 1, device=dev, dtype=torch.float64)], 1)
            pred = Fte @ W; yte = tgt[ntr:]
            return (1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum()).item()
        inp_r2.append(r2(inp)); post_r2.append(r2(post))
        print(f"[seed {seed}] target-pos R²: input {inp_r2[-1]:.3f}  post-attn {post_r2[-1]:.3f}")
    def ms(v):
        t = torch.tensor(v); return [round(t.mean().item(), 3), round(t.std().item(), 3)]
    result = {"target_pos_R2_input": ms(inp_r2), "target_pos_R2_postattn": ms(post_r2)}
    with open(f"{VOL_PATH}/vla_bond_seeds.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def vla_counterfactual(steps: int = 3000, seeds: int = 3):
    """G4/G9: a WORKING causal intervention on the policy with quantitative calibration.
    Keep the image+state fixed, but rewrite the instruction to name a DIFFERENT present
    object → the correct target moves. Does the predicted action follow the NEW target?
    Report R² of observed vs predicted counterfactual action (predicted = (new_target −
    grip)/H) and switch-rate (action moved to the new target's side). Clean cause→effect
    (change the named object) with calibration, and an open-loop behavioural proxy."""
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.synth_vla import make_batch, VOCAB, COLOR_TOK0, SHAPE_TOK0
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"
    H = 4
    def r2(obs, pred):
        return round((1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean(0)) ** 2).sum()).item(), 3)

    orig_r2, cf_r2, switch = [], [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        train_pool = make_batch(8192, dev)
        test = make_batch(2048, dev, k_objects=3)
        cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                        vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=4,
                        dim=256, n_layers=6, n_heads=8, action_horizon=4, action_dim=7)
        model = ChiVLA(cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        tccfg = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)
        model.train()
        for step in range(steps + 1):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, tccfg)
            idx = torch.randint(8192, (256,), device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(train_pool["img"][idx], train_pool["instr"][idx],
                                train_pool["state"][idx], train_pool["embodiment"][idx],
                                target_actions=train_pool["actions"][idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()

        grip = test["state"][:, :2]
        tgt_i = test["tgt_idx"]
        cf_i = (tgt_i + 1) % 3                                   # a different present object
        B = test["img"].shape[0]
        ar = torch.arange(B, device=dev)
        cf_pos = test["all_pos"][ar, cf_i]                       # (B,2) new target
        tgt_pos = test["all_pos"][ar, tgt_i]
        # counterfactual instruction: name object cf_i
        instr_cf = test["instr"].clone()
        instr_cf[:, 2] = COLOR_TOK0 + test["obj_pairs"][ar, cf_i, 0]
        instr_cf[:, 3] = SHAPE_TOK0 + test["obj_pairs"][ar, cf_i, 1]

        with torch.no_grad():
            a_orig, _ = model(test["img"], test["instr"], test["state"], test["embodiment"])
            a_cf, _ = model(test["img"], instr_cf, test["state"], test["embodiment"])
        obs_orig = a_orig[:, :, :2].mean(1)                      # (B,2)
        obs_cf = a_cf[:, :, :2].mean(1)
        pred_orig = (tgt_pos - grip) / H
        pred_cf = (cf_pos - grip) / H
        orig_r2.append(r2(obs_orig, pred_orig))
        cf_r2.append(r2(obs_cf, pred_cf))
        # switch-rate: did the counterfactual action move to the new target's side?
        d_new = (obs_cf - pred_cf).norm(dim=1)
        d_old = (obs_cf - pred_orig).norm(dim=1)
        switch.append(round((d_new < d_old).float().mean().item(), 3))
        print(f"[seed {seed}] orig_R2 {orig_r2[-1]} cf_R2 {cf_r2[-1]} switch {switch[-1]}")

    def ms(v):
        t = torch.tensor(v, dtype=torch.float64); return [round(t.mean().item(), 3), round(t.std().item(), 3)]
    result = {"original_action_R2": ms(orig_r2), "counterfactual_action_R2": ms(cf_r2),
              "switch_rate": ms(switch),
              "note": "rewrite instruction to name a different present object; does action follow the predicted new target"}
    with open(f"{VOL_PATH}/vla_counterfactual.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def topology_matched(epochs: int = 12):
    """G3: isolate topology from capability. Sweep width per architecture → collect
    (accuracy, atom-locality, params). If local-topology (conv/local-spatial) has higher
    coherence than dense AT MATCHED ACCURACY (coherence-vs-accuracy Pareto), the coherence
    gain is due to topology, not capacity — killing the 'conv-spatial just wins everything'
    confound. SVHN, exact quadratic pipeline."""
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_conv import ShallowBilinear
    from xvla.train.topology import class_quadratics, atom_report

    tl, vl = _svhn_loaders(flatten=False)
    vol.commit()
    in_ch, hw = 3, 32
    archs = [("dense", "global"), ("conv", "spatial"), ("local", "spatial")]
    widths = [8, 16, 32, 64, 128]
    rows = []
    for mode, readout in archs:
        for w in widths:
            torch.manual_seed(0)
            model = ShallowBilinear(mode=mode, readout=readout, grid=8, width=w, kernel=5,
                                    in_ch=in_ch, hw=hw, num_classes=10)
            lr = 1e-3 if mode == "local" else 2e-3
            acc = _train_shallow(model, tl, vl, epochs, lr=lr)
            params = model.num_params()
            Q, _, _ = class_quadratics(model, "cuda")
            al, rl = atom_report(Q, in_ch, hw, "cuda")
            name = mode if mode == "dense" else f"{mode}-{readout}"
            rows.append({"arch": name, "width": w, "acc": round(acc, 4),
                         "params_M": round(params / 1e6, 4),
                         "locality_ratio": round(al / rl, 3)})
            print(f"[{name} w{w}] acc {acc:.4f} params {params/1e6:.3f}M locality {al/rl:.2f}×")
            del Q; torch.cuda.empty_cache()

    with open(f"{VOL_PATH}/topology_matched.json", "w") as f:
        json.dump(rows, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(rows, indent=2))
    return rows


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def interp_baselines(epochs: int = 15, seeds: int = 3):
    """G1 (review-killer): compare the ODT global-Gram ranking against real baselines,
    not just random/local-SVD. On the χ-ViT patch bond (downstream crosses all attention):
    rank bond directions by (a) ODT output-sensitivity Gram [ours], (b) PCA of activations
    [unsupervised], (c) random; score each by deletion & insertion AUC of model accuracy,
    and by a linear-probe curve (class info in the top-k subspace). mean±std over seeds."""
    _bootstrap()
    import json
    import torch
    from xvla.models.vit import ViTConfig
    from xvla.train.train_vit import ViTTrainConfig, train_vit
    from xvla.train.odt import random_projector

    tl, vl = _svhn_loaders()
    vol.commit()
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    agg = {m: {"deletion_auc": [], "insertion_auc": [], "probe_auc": []}
           for m in ["odt", "pca", "random"]}
    accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        mcfg = ViTConfig(image_size=32, patch_size=4, dim=128, n_layers=4, n_heads=4,
                         num_classes=10, norm="per_token", qk_norm="per_token", attn="bilinear")
        h = train_vit(mcfg, ViTTrainConfig(epochs=epochs), tl, vl)
        model = h["model"].cuda().eval(); accs.append(h["best_acc"]); D = mcfg.dim

        def bond(imgs):
            x = model.patch(imgs).flatten(2).transpose(1, 2) + model.pos_emb
            return x

        def from_bond(x, P):
            x = x @ P.T
            xo = model.norm_out(model.blocks(x))
            return model.head(xo.mean(dim=1))

        xs, ys = [], []
        for imgs, labels in vl:
            xs.append(imgs); ys.append(labels)
            if sum(t.shape[0] for t in xs) >= 4000:
                break
        X = torch.cat(xs)[:4000].cuda(); Y = torch.cat(ys)[:4000].cuda()

        # (a) ODT output-sensitivity Gram
        G = torch.zeros(D, D, device="cuda", dtype=torch.float64); n = 0
        for i in range(0, 2000, 200):
            hb = bond(X[i:i+200]).detach().requires_grad_(True)
            sel = from_bond(hb, torch.eye(D, device="cuda")).gather(1, Y[i:i+200, None]).sum()
            g, = torch.autograd.grad(sel, hb)
            g = g.reshape(-1, D).double(); G += g.T @ g; n += g.shape[0]
        V_odt = torch.linalg.eigh(G / n)[1].flip(1)
        # (b) PCA of bond activations
        H = torch.cat([bond(X[i:i+500]).detach().reshape(-1, D) for i in range(0, 4000, 500)]).double()
        Hm = H.mean(0); cov = (H - Hm).T @ (H - Hm) / H.shape[0]
        V_pca = torch.linalg.eigh(cov)[1].flip(1)
        # (c) random
        rng = torch.Generator(device="cuda").manual_seed(seed)
        V_rand = torch.linalg.qr(torch.randn(D, D, generator=rng, device="cuda", dtype=torch.float64))[0]

        @torch.no_grad()
        def acc(P):
            c = t = 0
            for i in range(0, X.shape[0], 500):
                lg = from_bond(bond(X[i:i+500]), P.float())
                c += (lg.argmax(-1) == Y[i:i+500]).sum().item(); t += lg.shape[0]
            return c / t

        @torch.no_grad()
        def probe(Vk):
            Xp = torch.cat([bond(X[i:i+500]).mean(1).detach() for i in range(0, X.shape[0], 500)]).double()
            F = Xp @ Vk; ntr = int(0.7 * F.shape[0])
            Fa = torch.cat([F[:ntr], torch.ones(ntr, 1, device="cuda", dtype=torch.float64)], 1)
            Yt = torch.zeros(ntr, 10, device="cuda", dtype=torch.float64); Yt[torch.arange(ntr), Y[:ntr]] = 1
            W = torch.linalg.solve(Fa.T @ Fa + 1e-2 * torch.eye(Fa.shape[1], device="cuda", dtype=torch.float64), Fa.T @ Yt)
            Fte = torch.cat([F[ntr:], torch.ones(F.shape[0]-ntr, 1, device="cuda", dtype=torch.float64)], 1)
            return ((Fte @ W).argmax(1) == Y[ntr:]).float().mean().item()

        fr = [k / D for k in ks]
        def auc(ys_):
            return sum((fr[i+1]-fr[i])*(ys_[i]+ys_[i+1])/2 for i in range(len(fr)-1)) / (fr[-1]-fr[0])
        for name, V in [("odt", V_odt), ("pca", V_pca), ("random", V_rand)]:
            dele = [acc(torch.eye(D, device="cuda", dtype=torch.float64) - V[:, :k] @ V[:, :k].T) for k in ks]
            ins = [acc(V[:, :k] @ V[:, :k].T) for k in ks]
            prb = [probe(V[:, :k]) for k in ks]
            agg[name]["deletion_auc"].append(auc(dele))     # lower=better (drop-top kills acc)
            agg[name]["insertion_auc"].append(auc(ins))     # higher=better
            agg[name]["probe_auc"].append(auc(prb))
        del G, H, cov; torch.cuda.empty_cache()

    def ms(v):
        t = torch.tensor(v); return [round(t.mean().item(), 3), round(t.std().item(), 3)]
    result = {"vit_acc": ms(accs),
              **{m: {k: ms(v) for k, v in d.items()} for m, d in agg.items()},
              "note": "deletion_auc lower=better; insertion_auc & probe_auc higher=better"}
    with open(f"{VOL_PATH}/interp_baselines.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def vla_steering(steps: int = 3000):
    """Causal steering (Q2 payoff): is the nameable 'target-position' mechanism at the
    post-attention bond *causally* responsible for the action? We patch each sample's
    projection onto the 2D target-position subspace with a partner sample's, re-run the
    action head, and measure how far the predicted action moves toward the partner's
    action (transfer fraction). Compared to patching a RANDOM 2D subspace. High transfer
    for the target subspace, ~0 for random ⇒ the extracted mechanism causally controls
    the action — interpretability that predicts intervention, not just correlates."""
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.synth_vla import make_batch, VOCAB
    from xvla.train.train_lm import _lr_at, TrainConfig

    torch.manual_seed(0)
    dev = "cuda"
    train_pool = make_batch(8192, dev)
    probe_pool = make_batch(3000, dev)
    cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                    vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=4,
                    dim=256, n_layers=6, n_heads=8, action_horizon=4, action_dim=7)
    model = ChiVLA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    tccfg = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)

    def sample(pool, bs=256):
        idx = torch.randint(pool["img"].shape[0], (bs,), device=dev)
        return (pool["img"][idx], pool["instr"][idx], pool["state"][idx],
                pool["embodiment"][idx], pool["actions"][idx])

    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tccfg)
        img, instr, state, emb, act = sample(train_pool)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(img, instr, state, emb, target_actions=act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  step {step} mse {loss.item():.5f}")
    model.eval()
    H, D = cfg.action_horizon, cfg.dim

    @torch.no_grad()
    def aq_bond(img, instr, state, emb):
        B = img.shape[0]
        vis = model._visual_tokens(img)
        x = torch.cat([vis, model.bos.expand(B, -1, -1), model.tok_emb(instr),
                       model.state_proj(state)[:, None], model.embodiment_emb(emb)[:, None],
                       model.action_queries.expand(B, -1, -1)], dim=1)
        x = x + model.pos_emb[:, :x.shape[1]]
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        xo = model.norm_out(model.backbone(x, mask=mask))
        return xo[:, -H:]                                   # (B, H, d) post-attention

    p = probe_pool
    aq = torch.cat([aq_bond(p["img"][i:i+512], p["instr"][i:i+512], p["state"][i:i+512],
                            p["embodiment"][i:i+512]) for i in range(0, p["img"].shape[0], 512)]).double()
    base_act = model.action_head(aq.float()).double()       # (N,H,7)
    N = aq.shape[0]
    tgt = torch.stack([p["state"][:, 0] + H * p["actions"][:, 0, 0],
                       p["state"][:, 1] + H * p["actions"][:, 0, 1]], 1).double()  # (N,2)

    # target-position subspace at the bond: ridge aq(pooled) -> target, orthonormalize weights
    aqm = aq.mean(1)                                        # (N, d)
    Fa = torch.cat([aqm, torch.ones(N, 1, device=dev, dtype=torch.float64)], 1)
    W = torch.linalg.solve(Fa.T @ Fa + 1e-2 * torch.eye(D + 1, device=dev, dtype=torch.float64),
                           Fa.T @ tgt)[:D]                  # (d, 2)
    S_tgt = torch.linalg.qr(W)[0]                           # (d, 2) orthonormal
    g = torch.Generator(device=dev).manual_seed(0)
    S_rnd = torch.linalg.qr(torch.randn(D, 2, generator=g, device=dev, dtype=torch.float64))[0]
    perm = torch.randperm(N, generator=g, device=dev)

    @torch.no_grad()
    def transfer(S):
        proj = aq @ S                                       # (N,H,2)
        patched = aq - proj @ S.T + (aq[perm] @ S) @ S.T    # swap S-projection with partner's
        new_act = model.action_head(patched.float()).double()
        d_act = (new_act - base_act)[..., :2].reshape(N, -1)          # position dims
        target_dir = (base_act[perm] - base_act)[..., :2].reshape(N, -1)
        num = (d_act * target_dir).sum(1)
        den = (target_dir * target_dir).sum(1).clamp_min(1e-9)
        return round((num / den).mean().item(), 3)         # fraction moved toward partner

    r2 = round((1 - ((tgt - Fa @ torch.cat([W, torch.zeros(1, 2, device=dev, dtype=torch.float64)]))
                     ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()).item(), 3)
    result = {"target_decode_R2": r2,
              "transfer_target_subspace": transfer(S_tgt),
              "transfer_random_subspace": transfer(S_rnd)}
    with open(f"{VOL_PATH}/vla_steering.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def topology_depth(epochs: int = 15, seeds: int = 2):
    """Experiment A-depth (Q1 at depth): does topology→coherence hold in DEEP bilinear
    nets? Dense (ChiMLP, flatten) vs conv (ConvBilinearDeep, local+spatial), depth 1 & 3,
    on SVHN + CIFAR, analyzed via the data-driven input-space Gram (works at any depth).
    Reports mean±std of accuracy, input-atom locality ratio, and global-vs-random input
    truncation. depth=1 also cross-checks the data-driven method against Exp A's exact Q."""
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
    from xvla.models.chi_conv import ConvBilinearDeep
    from xvla.train.topology import input_analysis

    agg = {}
    for ds in ["svhn", "cifar"]:
        tl, vl = (_svhn_loaders(flatten=False) if ds == "svhn" else _cifar_loaders())
        vol.commit()
        xs, ys = [], []
        for imgs, labels in vl:
            xs.append(imgs); ys.append(labels)
            if sum(t.shape[0] for t in xs) >= 2000:
                break
        X = torch.cat(xs)[:2000].cuda(); Y = torch.cat(ys)[:2000].cuda()
        for arch in ["dense", "conv"]:
            for depth in [1, 3]:
                accs, ratios, g64, r64 = [], [], [], []
                for seed in range(seeds):
                    torch.manual_seed(seed)
                    if arch == "dense":
                        model = ChiMLP(ChiMLPConfig(in_dim=3072, dim=48, n_layers=depth,
                                                    num_classes=10, norm="scalar_rbn"))
                        f = lambda z: model(z)[0]
                    else:
                        model = ConvBilinearDeep(depth=depth, width=48, kernel=5, grid=8,
                                                 in_ch=3, hw=32, num_classes=10)
                        f = lambda z: model.logits(z)
                    acc = _train_shallow(model, tl, vl, epochs, lr=2e-3)
                    model.cuda().eval()
                    res = input_analysis(f, X, Y, "cuda")
                    accs.append(acc); ratios.append(res["locality_ratio"])
                    g64.append(res["global_curve"][64]); r64.append(res["random_curve"][64])
                    torch.cuda.empty_cache()

                def ms(v):
                    t = torch.tensor(v)
                    return [round(t.mean().item(), 3), round(t.std().item(), 3)]
                key = f"{ds}/{arch}-d{depth}"
                agg[key] = {"acc": ms(accs), "locality_ratio": ms(ratios),
                            "global@64": ms(g64), "random@64": ms(r64)}
                print(f"[{key}] acc {ms(accs)} locality_ratio {ms(ratios)} "
                      f"global@64 {ms(g64)} random@64 {ms(r64)}")

    with open(f"{VOL_PATH}/topology_depth.json", "w") as f:
        json.dump(agg, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(agg, indent=2))
    return agg


@app.function(image=image, gpu="A100", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def topology_derisk(epochs: int = 12, seeds: int = 3):
    """De-risk Exp A across DATASETS (SVHN, CIFAR) and SEEDS (Q1/Q4 robustness).

    Reuses the exact single-layer quadratic-form pipeline (tested), for the three
    architectures that trained well and showed the effect. Reports mean±std over
    seeds of accuracy, atom-locality ratio, and the faithfulness gap. (Depth is a
    separate run — deep bilinear needs foldable-norm stability engineering.)
    """
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_conv import ShallowBilinear
    from xvla.train.topology import (class_quadratics, quad_logits, atom_report,
                                     faithfulness)

    in_ch, hw = 3, 32
    D = in_ch * hw * hw
    archs = [("dense", 5, "global"), ("conv", 5, "spatial"), ("local", 5, "spatial")]
    datasets = ["svhn", "cifar"]
    agg = {}
    for ds in datasets:
        tl, vl = (_svhn_loaders(flatten=False) if ds == "svhn" else _cifar_loaders())
        vol.commit()
        xs, ys = [], []
        for imgs, labels in vl:
            xs.append(imgs); ys.append(labels)
            if sum(t.shape[0] for t in xs) >= 4000:
                break
        x = torch.cat(xs)[:4000].cuda().double().reshape(-1, D)
        y = torch.cat(ys)[:4000].cuda()
        for mode, k, readout in archs:
            name = mode if mode == "dense" else f"{mode}-{readout}"
            accs, ratios, dtop, drand = [], [], [], []
            for seed in range(seeds):
                torch.manual_seed(seed)
                model = ShallowBilinear(mode=mode, readout=readout, grid=8, width=48,
                                        kernel=k, in_ch=in_ch, hw=hw, num_classes=10)
                lr = 1e-3 if mode == "local" else 2e-3
                acc = _train_shallow(model, tl, vl, epochs, lr=lr)
                Q, l, a = class_quadratics(model, "cuda")
                al, rl = atom_report(Q, in_ch, hw, "cuda")
                f = faithfulness(Q, l, a, x, y, [16, 64], "cuda")
                accs.append(acc); ratios.append(al / rl)
                dtop.append(f["drop_top"][64]); drand.append(f["drop_random"][64])
                del Q; torch.cuda.empty_cache()
            def ms(v):
                t = torch.tensor(v)
                return [round(t.mean().item(), 3), round(t.std().item(), 3)]
            agg[f"{ds}/{name}"] = {"acc_mean_std": ms(accs),
                                   "locality_ratio_mean_std": ms(ratios),
                                   "drop_top64_mean_std": ms(dtop),
                                   "drop_random64_mean_std": ms(drand)}
            print(f"[{ds}/{name}] acc {ms(accs)} locality_ratio {ms(ratios)} "
                  f"drop_top64 {ms(dtop)} drop_rand64 {ms(drand)}")

    with open(f"{VOL_PATH}/topology_derisk.json", "w") as f:
        json.dump(agg, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(agg, indent=2))
    return agg


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def topology_sweep(epochs: int = 12):
    """Experiment A (Q1/Q4): does TOPOLOGY, not convertibility, drive coherence?

    Shallow single-bilinear SVHN classifiers whose class logit is exactly a
    quadratic form in the pixels. Same ODT-style analysis on each; only the
    linL/linR topology differs. For every architecture: accuracy, params, exact
    reconstruction, eigen-atom locality vs random (coherence), and the drop-top vs
    drop-random faithfulness battery (causality). Prediction: conv locality ≫
    dense; peaks then falls with kernel K; `local` (unshared) isolates weight-
    sharing from mere locality.
    """
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_conv import ShallowBilinear
    from xvla.train.topology import (class_quadratics, quad_logits, atom_report,
                                     faithfulness)

    tl, vl = _svhn_loaders(flatten=False)
    vol.commit()
    in_ch, hw = 3, 32
    D = in_ch * hw * hw

    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 4000:
            break
    x = torch.cat(xs)[:4000].cuda().double().reshape(-1, D)
    y = torch.cat(ys)[:4000].cuda()
    ranks = [1, 2, 4, 8, 16, 32, 64]

    # (mode, kernel, readout): dense baseline; conv/local with global vs spatial
    # readout. `global` conv is translation-invariant + capacity-starved; `spatial`
    # restores capacity and breaks translation-invariance — the fair comparison.
    archs = [("dense", 5, "global"),
             ("conv", 5, "global"), ("conv", 5, "spatial"),
             ("local", 5, "global"), ("local", 5, "spatial")]
    results = {}
    for mode, k, readout in archs:
        tag = mode if mode == "dense" else f"{mode}-K{k}-{readout}"
        name = tag
        torch.manual_seed(0)
        model = ShallowBilinear(mode=mode, readout=readout, grid=8, width=48, kernel=k,
                                in_ch=in_ch, hw=hw, num_classes=10)
        lr = 1e-3 if mode == "local" else 2e-3
        acc = _train_shallow(model, tl, vl, epochs, lr=lr)
        params = model.num_params()

        Q, l, a = class_quadratics(model, "cuda")
        # exact reconstruction of the quadratic form vs the module
        with torch.no_grad():
            ref = model.double().logits(x.reshape(-1, in_ch, hw, hw))
        recon = (ref - quad_logits(Q, l, a, x)).abs().max().item()
        quad_acc = (quad_logits(Q, l, a, x).argmax(-1) == y).float().mean().item()
        atom_loc, rand_loc = atom_report(Q, in_ch, hw, "cuda")
        faith = faithfulness(Q, l, a, x, y, ranks, "cuda")

        results[name] = {
            "mode": mode, "kernel": k, "readout": readout, "val_acc": round(acc, 4),
            "params_M": round(params / 1e6, 4), "recon_max_abs": recon,
            "quad_acc": round(quad_acc, 4),
            "atom_locality": round(atom_loc, 4), "random_locality": round(rand_loc, 4),
            "locality_ratio": round(atom_loc / rand_loc, 3),
            "faithfulness": faith,
        }
        print(f"[{name}] acc {acc:.4f} params {params/1e6:.3f}M recon {recon:.1e} "
              f"locality {atom_loc:.3f} (ratio {atom_loc/rand_loc:.2f}×) "
              f"drop_top@16 {faith['drop_top'][16]:.3f} drop_rand@16 {faith['drop_random'][16]:.3f}")
        del Q; torch.cuda.empty_cache()

    with open(f"{VOL_PATH}/topology_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    summary = {n: {"acc": r["val_acc"], "params_M": r["params_M"],
                   "locality_ratio": r["locality_ratio"],
                   "drop_top@16": r["faithfulness"]["drop_top"][16],
                   "drop_random@16": r["faithfulness"]["drop_random"][16]}
               for n, r in results.items()}
    print("RESULT:", json.dumps(summary, indent=2))
    return results


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def odt_vla_bond(steps: int = 3000):
    """Experiment C-v2 (Q2 completion): is the policy nameable at the RIGHT bond?

    C-v1 found action mechanisms are not linearly readable at the backbone INPUT
    bond (they're built downstream by attention). Here we linearly probe semantic
    quantities at the input bond vs the POST-attention action-query bond:
      * grip x/y (in the state token) — should decode at both,
      * target color/shape (in the instruction tokens, linear embed) — both,
      * TARGET position x/y (the *bound* quantity: needs attention to bind the named
        identity to its location) — should be LOW at input, HIGH post-attention.
    If so, the classifier recipe DOES work for policies at a post-attention bond, and
    the mechanism (target-position) is genuinely nameable there.
    """
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.synth_vla import (make_batch, VOCAB, COLOR_TOK0, SHAPE_TOK0,
                                      N_COLORS, N_SHAPES)
    from xvla.train.train_lm import _lr_at, TrainConfig

    torch.manual_seed(0)
    dev = "cuda"
    train_pool = make_batch(8192, dev)
    probe_pool = make_batch(3000, dev)
    cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                    vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=4,
                    dim=256, n_layers=6, n_heads=8, action_horizon=4, action_dim=7)
    model = ChiVLA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    tccfg = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)

    def sample(pool, bs=256):
        idx = torch.randint(pool["img"].shape[0], (bs,), device=dev)
        return (pool["img"][idx], pool["instr"][idx], pool["state"][idx],
                pool["embodiment"][idx], pool["actions"][idx])

    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tccfg)
        img, instr, state, emb, act = sample(train_pool)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(img, instr, state, emb, target_actions=act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  step {step} mse {loss.item():.5f}")
    model.eval()

    H = cfg.action_horizon

    @torch.no_grad()
    def bonds(img, instr, state, emb):
        B = img.shape[0]
        vis = model._visual_tokens(img)
        x = torch.cat([vis, model.bos.expand(B, -1, -1), model.tok_emb(instr),
                       model.state_proj(state)[:, None],
                       model.embodiment_emb(emb)[:, None],
                       model.action_queries.expand(B, -1, -1)], dim=1)
        x = x + model.pos_emb[:, :x.shape[1]]
        inp = x.mean(dim=1)                                     # input-bond feature
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        xo = model.norm_out(model.backbone(x, mask=mask))
        out = xo[:, -H:].mean(dim=1)                           # post-attention action-query
        return inp.double(), out.double()

    # features + ground-truth targets on the probe pool
    p = probe_pool
    inp_f, out_f = [], []
    for i in range(0, p["img"].shape[0], 512):
        sl = slice(i, i + 512)
        a, b = bonds(p["img"][sl], p["instr"][sl], p["state"][sl], p["embodiment"][sl])
        inp_f.append(a); out_f.append(b)
    inp_f = torch.cat(inp_f); out_f = torch.cat(out_f)
    N = inp_f.shape[0]
    tgt_x = (p["state"][:, 0] + H * p["actions"][:, 0, 0]).double()
    tgt_y = (p["state"][:, 1] + H * p["actions"][:, 0, 1]).double()
    grip_x = p["state"][:, 0].double(); grip_y = p["state"][:, 1].double()
    tgt_col = (p["instr"][:, 2] - COLOR_TOK0).long()
    tgt_shape = (p["instr"][:, 3] - SHAPE_TOK0).long()

    ntr = int(0.7 * N)

    def ridge_r2(F, y):
        Ftr, Fte = F[:ntr], F[ntr:]; ytr, yte = y[:ntr], y[ntr:]
        Fa = torch.cat([Ftr, torch.ones(ntr, 1, device=dev, dtype=torch.float64)], 1)
        W = torch.linalg.solve(Fa.T @ Fa + 1e-2 * torch.eye(Fa.shape[1], device=dev, dtype=torch.float64),
                               Fa.T @ ytr[:, None])
        Fte_a = torch.cat([Fte, torch.ones(N - ntr, 1, device=dev, dtype=torch.float64)], 1)
        pred = (Fte_a @ W)[:, 0]
        ss_res = ((yte - pred) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum()
        return (1 - ss_res / ss_tot).item()

    def lin_acc(F, y, ncls):
        Ftr, Fte = F[:ntr], F[ntr:]; ytr, yte = y[:ntr], y[ntr:]
        Y = torch.zeros(ntr, ncls, device=dev, dtype=torch.float64); Y[torch.arange(ntr), ytr] = 1
        Fa = torch.cat([Ftr, torch.ones(ntr, 1, device=dev, dtype=torch.float64)], 1)
        W = torch.linalg.solve(Fa.T @ Fa + 1e-2 * torch.eye(Fa.shape[1], device=dev, dtype=torch.float64),
                               Fa.T @ Y)
        Fte_a = torch.cat([Fte, torch.ones(N - ntr, 1, device=dev, dtype=torch.float64)], 1)
        return ((Fte_a @ W).argmax(1) == yte).float().mean().item()

    probe = {
        "target_x_R2":   {"input": round(ridge_r2(inp_f, tgt_x), 3),  "post_attn": round(ridge_r2(out_f, tgt_x), 3)},
        "target_y_R2":   {"input": round(ridge_r2(inp_f, tgt_y), 3),  "post_attn": round(ridge_r2(out_f, tgt_y), 3)},
        "grip_x_R2":     {"input": round(ridge_r2(inp_f, grip_x), 3), "post_attn": round(ridge_r2(out_f, grip_x), 3)},
        "grip_y_R2":     {"input": round(ridge_r2(inp_f, grip_y), 3), "post_attn": round(ridge_r2(out_f, grip_y), 3)},
        "target_color_acc": {"input": round(lin_acc(inp_f, tgt_col, N_COLORS), 3),  "post_attn": round(lin_acc(out_f, tgt_col, N_COLORS), 3)},
        "target_shape_acc": {"input": round(lin_acc(inp_f, tgt_shape, N_SHAPES), 3), "post_attn": round(lin_acc(out_f, tgt_shape, N_SHAPES), 3)},
    }
    with open(f"{VOL_PATH}/odt_vla_bond.json", "w") as f:
        json.dump(probe, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(probe, indent=2))
    return probe


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def odt_vla_action(steps: int = 3000):
    """Experiment C (Q2): what does "interpretable" mean for a POLICY?

    Trains the synthetic reach χ-VLA, then does ODT conditioned on each ACTION
    dimension at the backbone-input bond (downstream = full causal transformer +
    head). Tests three things that define an *action* coherence axis:
      (i)   action-resolved: top-k subspaces of the x-step vs y-step Grams differ
            (each action primitive selects its own mechanisms);
      (ii)  causal: truncating the bond onto that action's top-k GLOBAL directions
            keeps its MSE far better than a random k-subspace;
      (iii) nameable: the top x-mechanism's bond projection correlates with the
            ground-truth target x-position (recoverable as grip + H·step), i.e. the
            mechanism literally *is* "where is the target in x".
    """
    _bootstrap()
    import json
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.synth_vla import make_batch, VOCAB
    from xvla.train.train_lm import _lr_at, TrainConfig
    from xvla.train.odt import random_projector

    torch.manual_seed(0)
    dev = "cuda"
    train_pool = make_batch(8192, dev)
    test_pool = make_batch(2048, dev)
    cfg = VLAConfig(image_size=32, patch_size=4, vit_dim=128, vit_layers=3, vit_heads=8,
                    vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=4,
                    dim=256, n_layers=6, n_heads=8, action_horizon=4, action_dim=7)
    model = ChiVLA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    tccfg = TrainConfig(train_bin="", val_bin="", lr=1e-3, max_steps=steps, warmup_frac=0.05)

    def sample(pool, bs=256):
        idx = torch.randint(pool["img"].shape[0], (bs,), device=dev)
        return (pool["img"][idx], pool["instr"][idx], pool["state"][idx],
                pool["embodiment"][idx], pool["actions"][idx])

    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tccfg)
        img, instr, state, emb, act = sample(train_pool)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(img, instr, state, emb, target_actions=act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  step {step} mse {loss.item():.5f}")
    model.eval()
    D = cfg.dim

    def to_bond(img, instr, state, emb):
        B = img.shape[0]
        vis = model._visual_tokens(img)
        x = torch.cat([vis, model.bos.expand(B, -1, -1), model.tok_emb(instr),
                       model.state_proj(state)[:, None],
                       model.embodiment_emb(emb)[:, None],
                       model.action_queries.expand(B, -1, -1)], dim=1)
        return x + model.pos_emb[:, :x.shape[1]]

    def from_bond(x, P=None):
        if P is not None:
            x = x @ P.T
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        x = model.norm_out(model.backbone(x, mask=mask))
        return model.action_head(x[:, -cfg.action_horizon:])       # (B,H,7)

    # ---- action-conditioned Grams at the bond ----
    grams = {}
    for a in (0, 1, 6):
        G = torch.zeros(D, D, device=dev, dtype=torch.float64)
        n = 0
        for i in range(0, 1024, 256):
            img, instr, state, emb, _ = sample(test_pool, 256)
            xb = to_bond(img, instr, state, emb).detach().requires_grad_(True)
            act = from_bond(xb)
            gsel, = torch.autograd.grad(act[:, :, a].sum(), xb)
            g = gsel.reshape(-1, D).double()
            G += g.T @ g; n += g.shape[0]
        grams[a] = G / n

    def topk_proj(G, k):
        ev, V = torch.linalg.eigh(G)
        return V.flip(1)[:, :k]

    # ---- (ii) causal faithfulness per action dim ----
    tp = test_pool
    Xb = to_bond(tp["img"], tp["instr"], tp["state"], tp["embodiment"]).detach()
    Atrue = tp["actions"]
    ks = [1, 2, 4, 8, 16, 32, 64]
    rng = torch.Generator(device=dev).manual_seed(0)

    @torch.no_grad()
    def dim_mse(P, a):
        tot = n = 0.0
        for i in range(0, Xb.shape[0], 512):
            pred = from_bond(Xb[i:i + 512], P)
            tot += (pred[:, :, a] - Atrue[i:i + 512, :, a]).pow(2).sum().item()
            n += pred[:, :, a].numel()
        return tot / n

    faith = {}
    for a, name in [(0, "x"), (1, "y")]:
        Vg = topk_proj(grams[a], 64)
        curve_g, curve_r = {}, {}
        for k in ks:
            Pg = (Vg[:, :k] @ Vg[:, :k].T).float()
            curve_g[k] = dim_mse(Pg, a)
            curve_r[k] = dim_mse(random_projector(D, k, dev, rng).float(), a)
        faith[name] = {"global": curve_g, "random": curve_r}

    # ---- (i) action-resolution: subspace overlap x vs y ----
    def overlap(Ga, Gb, k):
        Va, Vb = topk_proj(Ga, k), topk_proj(Gb, k)
        return (torch.trace((Va @ Va.T) @ (Vb @ Vb.T)) / k).item()   # 0..1
    resolution = {k: {"x_vs_y": round(overlap(grams[0], grams[1], k), 3),
                      "random_baseline": round(k / D, 3)} for k in [4, 8, 16, 32]}

    # ---- (iii) nameable: top mechanism vs ground-truth target position ----
    H = cfg.action_horizon
    tgt_x = tp["state"][:, 0] + H * Atrue[:, 0, 0]
    tgt_y = tp["state"][:, 1] + H * Atrue[:, 0, 1]
    bm = Xb.mean(dim=1).double()                                     # (B, D) pooled bond
    def corr(v, t):
        p = (bm @ v)
        p = (p - p.mean()) / p.std().clamp_min(1e-9)
        t = (t.double() - t.double().mean()) / t.double().std().clamp_min(1e-9)
        return (p * t).mean().abs().item()
    vx, vy = topk_proj(grams[0], 1)[:, 0], topk_proj(grams[1], 1)[:, 0]
    naming = {
        "xmech_vs_targetX": round(corr(vx, tgt_x), 3),
        "xmech_vs_targetY": round(corr(vx, tgt_y), 3),
        "ymech_vs_targetY": round(corr(vy, tgt_y), 3),
        "ymech_vs_targetX": round(corr(vy, tgt_x), 3),
    }

    result = {"faithfulness": faith, "action_resolution": resolution, "naming": naming}
    with open(f"{VOL_PATH}/odt_vla_action.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def odt_vit(epochs: int = 15):
    """Experiment B (Q3): does causal + coherent ODT structure survive ATTENTION on
    the real softmax-free χ-ViT?

    A deep attention model is not a single quadratic, so we use the data-driven
    output-sensitivity Gram at the patch bond (as in Finding 19 for a char LM), whose
    downstream crosses every attention layer. We test (i) faithfulness: truncate the
    bond onto the top-k GLOBAL directions vs a RANDOM k-subspace and measure accuracy
    (global ≫ random ⇒ causal low-rank structure survives attention); (ii) coherence:
    project the top global directions through the patch embedding to pixel patches.
    """
    _bootstrap()
    import json
    import torch
    from xvla.models.vit import ViTConfig
    from xvla.train.train_vit import ViTTrainConfig, train_vit
    from xvla.train.odt import random_projector

    tl, vl = _svhn_loaders()
    vol.commit()
    mcfg = ViTConfig(image_size=32, patch_size=4, dim=128, n_layers=4, n_heads=4,
                     num_classes=10, norm="per_token", qk_norm="per_token", attn="bilinear")
    h = train_vit(mcfg, ViTTrainConfig(epochs=epochs), tl, vl)
    model = h["model"].cuda().eval()
    D = mcfg.dim
    print(f"χ-ViT dim={D} L={mcfg.n_layers}: acc {h['best_acc']:.4f}")

    def bond(imgs):
        x = model.patch(imgs).flatten(2).transpose(1, 2) + model.pos_emb
        if model.use_cls:
            x = torch.cat([model.cls.expand(imgs.shape[0], -1, -1), x], dim=1)
        return x

    def from_bond(x, P=None):
        if P is not None:
            x = x @ P.T
        x = model.blocks(x); x = model.norm_out(x)
        pooled = x[:, 0] if model.use_cls else x.mean(dim=1)
        return model.head(pooled)

    # gather calibration + test tensors
    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 4000:
            break
    X = torch.cat(xs)[:4000].cuda(); Y = torch.cat(ys)[:4000].cuda()

    # data-driven global Gram at the bond: G = E_tokens[ g gᵀ ], g = ∂(true-class logit)/∂h
    G = torch.zeros(D, D, device="cuda", dtype=torch.float64)
    ntok = 0
    for i in range(0, 2000, 200):
        imgs = X[i:i + 200]; y = Y[i:i + 200]
        hb = bond(imgs).detach().requires_grad_(True)
        logits = from_bond(hb)
        sel = logits.gather(1, y[:, None]).sum()
        g, = torch.autograd.grad(sel, hb)                    # (B, N, D)
        g = g.reshape(-1, D).double()
        G += g.T @ g; ntok += g.shape[0]
    G /= ntok
    evals, evecs = torch.linalg.eigh(G)
    evecs = evecs.flip(1)                                     # top global directions first

    @torch.no_grad()
    def acc(P):
        cor = tot = 0
        for i in range(0, X.shape[0], 500):
            lg = from_bond(bond(X[i:i + 500]), P)
            cor += (lg.argmax(-1) == Y[i:i + 500]).sum().item(); tot += 500
        return cor / tot

    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    rng = torch.Generator(device="cuda").manual_seed(0)
    glob, rand = {}, {}
    for k in ks:
        Vk = evecs[:, :k].double()
        glob[k] = acc((Vk @ Vk.T).float())
        rand[k] = acc(random_projector(D, k, "cuda", rng).float())
        print(f"  k={k:3d}  global {glob[k]:.3f}  random {rand[k]:.3f}")

    # coherence: top global directions → pixel patches through the patch embedding
    pw = model.patch.weight.detach().double()                # (D, 3, ps, ps)
    ps = mcfg.patch_size
    def patch_locality(v):
        pat = torch.einsum("d,dchw->chw", v, pw)             # (3, ps, ps)
        e = (pat ** 2).sum(0).flatten()
        kk = max(1, int(0.25 * e.numel()))
        return (torch.topk(e, kk).values.sum() / e.sum().clamp_min(1e-30)).item(), pat
    rand_loc = []
    for _ in range(200):
        v = torch.randn(D, device="cuda", dtype=torch.float64)
        rand_loc.append(patch_locality(v)[0])
    rand_loc = sum(rand_loc) / len(rand_loc)
    atoms, atom_loc = [], []
    for j in range(6):
        loc, pat = patch_locality(evecs[:, j].double())
        atom_loc.append(loc); atoms.append(pat.float())
    _save_atom_grid([atoms], f"{VOL_PATH}/odt_vit_atoms.png", row_labels=["patch"])
    atom_loc = sum(atom_loc) / len(atom_loc)

    result = {
        "acc": round(h["best_acc"], 4), "dim": D, "layers": mcfg.n_layers,
        "global_curve": glob, "random_curve": rand,
        "global_beats_random_at": [k for k in ks if glob[k] > rand[k] + 1e-4],
        "patch_atom_locality": round(atom_loc, 4),
        "patch_random_locality": round(rand_loc, 4),
        "patch_locality_ratio": round(atom_loc / rand_loc, 3),
    }
    with open(f"{VOL_PATH}/odt_vit.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def train_chi_vit(epochs: int = 30, norm: str = "per_token", dataset: str = "svhn",
                  attn: str = "bilinear"):
    """Milestone 2: train a χ-ViT (softmax-free) on SVHN/CIFAR; report accuracy.
    attn='softmax' gives the matched baseline for the §19 ≥90% gate."""
    _bootstrap()
    import json
    from xvla.models.vit import ViTConfig
    from xvla.train.train_vit import ViTTrainConfig, train_vit
    tl, vl = (_svhn_loaders() if dataset == "svhn" else _cifar_loaders())
    vol.commit()
    mcfg = ViTConfig(image_size=32, patch_size=4, dim=256, n_layers=6, n_heads=8,
                     num_classes=10, norm=norm, qk_norm=norm, attn=attn)
    h = train_vit(mcfg, ViTTrainConfig(epochs=epochs), tl, vl)
    result = {"dataset": dataset, "attn": attn, "norm": norm,
              "best_acc": h["best_acc"], "final_acc": h["final_acc"],
              "params_M": round(h["model"].num_params() / 1e6, 2)}
    with open(f"{VOL_PATH}/chi_vit_{dataset}_{attn}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def pretest_tail(steps: int = 1200):
    """P0 (panel-mandated, unanimous): measure the activation tail ρ = max_i(r_i)/c
    at every norm site, for BILINEAR vs SOFTMAX attention. Tests whether the
    scalar calibration constant c covers the deployment tail — i.e. whether the
    fat-tail fear (a softmax sum-to-one/register-token artifact) even applies to
    softmax-free bilinear attention.

    Verdict per arch: THIN if max ρ < ~3 and no depth growth (scalar c covers the
    tail → W2 unnecessary); FAT if max ρ > ~10 or grows with depth (→ build W2)."""
    _bootstrap()
    import json, math
    from collections import defaultdict
    import torch
    from xvla.models.lm import ChiLanguageModel, LMConfig
    from xvla.nn.normalization import PerTokenRmsNorm
    from xvla.nn.attention import BilinearAttention
    from xvla.train.data import prepare_tinyshakespeare, TokenBinDataset
    from xvla.train.train_lm import TrainConfig, _lr_at

    train_bin, val_bin, vocab = prepare_tinyshakespeare(f"{VOL_PATH}/tinyshakespeare")
    vol.commit()
    tc = TrainConfig(train_bin=train_bin, val_bin=val_bin, seq_len=256, batch_size=32,
                     lr=6e-4, max_steps=steps, warmup_frac=0.05)
    ds = TokenBinDataset(train_bin, tc.seq_len)
    val_ds = TokenBinDataset(val_bin, tc.seq_len)

    out = {}
    for arch in ("bilinear", "softmax"):
        torch.manual_seed(0)
        cfg = LMConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=12, max_seq_len=256,
                       attn=arch, ffn="bilinear", norm="per_token", qk_norm="per_token")
        model = ChiLanguageModel(cfg).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=(0.9, 0.95),
                                weight_decay=tc.weight_decay)
        model.train()
        for step in range(tc.max_steps + 1):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, tc)
            opt.zero_grad(set_to_none=True)
            x, y = ds.batch(tc.batch_size, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"[{arch}] trained, final loss {loss.item():.3f}")

        # Per-site tail stats: r_i = per-token RMS of the norm-site input.
        stats = defaultdict(lambda: {"max": 0.0, "sum": 0.0, "cnt": 0.0})

        def rec(nm, r):
            s = stats[nm]
            s["max"] = max(s["max"], r.max().item())
            s["sum"] += r.sum().item(); s["cnt"] += r.numel()

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, PerTokenRmsNorm):     # residual-stream (FFN) sites
                def h(mod, inp, nm=name):
                    x0 = inp[0]
                    rec(nm, x0.float().pow(2).mean(-1).sqrt())
                hooks.append(m.register_forward_pre_hook(h))
            if isinstance(m, BilinearAttention):    # QK sites (attention crux)
                def h(mod, inp, nm=name):
                    x0 = inp[0]; B, N, _ = x0.shape
                    q = mod.wq1(x0).view(B, N, mod.n_heads, mod.head_dim)
                    rec(nm + ".q1", q.float().pow(2).mean(-1).sqrt())
                hooks.append(m.register_forward_pre_hook(h))

        model.eval()
        with torch.no_grad():
            for _ in range(200):                    # large sweep for the tail
                xb, _ = val_ds.batch(tc.batch_size, "cuda")
                model(xb)
        for hd in hooks:
            hd.remove()

        rho = {nm: s["max"] / (s["sum"] / s["cnt"]) for nm, s in stats.items()}
        max_rho = max(rho.values())
        verdict = "THIN" if max_rho < 3.0 else ("MODERATE" if max_rho < 10.0 else "FAT")
        out[arch] = {"final_loss": loss.item(), "max_rho": max_rho, "verdict": verdict,
                     "per_site_rho": {k: round(v, 2) for k, v in sorted(rho.items())}}
        print(f"[{arch}] max ρ = {max_rho:.2f}  → {verdict}")
        for k, v in sorted(rho.items()):
            print(f"    {k:34s} ρ={v:.2f}")

    with open(f"{VOL_PATH}/pretest_tail.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({a: {"max_rho": round(o["max_rho"], 2), "verdict": o["verdict"]}
                                 for a, o in out.items()}, indent=2))
    return out


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
