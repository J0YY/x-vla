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
        ignore=["*.pdf", "__pycache__", "*.pyc", ".git", ".venv", "data", "out",
                "tmp", "tmp/**"],
    )
)

app = modal.App("xvla")
vol = modal.Volume.from_name("xvla-data", create_if_missing=True)
VOL_PATH = "/vol"

# ---- LIBERO closed-loop simulator base (MuJoCo + robosuite + LIBERO, headless EGL) ----
# NOTE: build steps only; add_local_dir must come LAST (Modal forbids build steps after it),
# so both libero_image and openvla_image derive from this base and add locals themselves.
_libero_base = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-dev", "libglib2.0-0", "libosmesa6-dev",
                 "libegl1-mesa-dev", "libgles2-mesa-dev", "libglfw3", "libglew-dev",
                 "patchelf", "gcc", "g++")
    .pip_install("torch>=2.2", "torchvision", "numpy<2", "scipy", "pillow",
                 "huggingface_hub", "datasets", "opencv-python-headless",
                 "mujoco==3.1.6", "robosuite==1.4.1", "bddl", "easydict",
                 "termcolor", "thop", "cloudpickle", "gym==0.25.2", "hydra-core", "pyyaml",
                 "matplotlib", "imageio", "imageio-ffmpeg", "future", "scikit-learn")
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "cd /opt/LIBERO && pip install --no-deps -e .",
        # pre-write LIBERO's config so first import doesn't prompt interactively (EOFError)
        "mkdir -p /root/.libero",
        "python3 -c \"import os,yaml; root='/opt/LIBERO'; f=lambda n:next((d for d,s,fs in os.walk(root) if os.path.basename(d)==n),''); b=f('bddl_files'); i=f('init_files'); a=f('assets'); yaml.safe_dump({'benchmark_root':os.path.dirname(b),'bddl_files':b,'init_states':i,'assets':a,'datasets':root+'/datasets'}, open('/root/.libero/config.yaml','w')); print(open('/root/.libero/config.yaml').read())\"")
    .env({"PYTHONPATH": "/opt/LIBERO",
          "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0",
          "TOKENIZERS_PARALLELISM": "false"})
)

_LOCAL = dict(
    ignore=[
        "*.pdf", "__pycache__", "**/__pycache__", "*.pyc", "**/*.pyc",
        ".git", ".venv", "data", "out",
        # tmp/ is local scratch (venv, extracted PDFs, agent workspaces, run logs).
        # Nothing remote reads it, it is ~3GB, and writing a log there mid-build
        # aborts the build with "modified during build process".
        "tmp", "tmp/**",
    ]
)
libero_image = _libero_base.add_local_dir(".", PROJ, **_LOCAL)

# ---- OpenVLA teacher image: LIBERO base + HF transformers stack, then locals LAST ----
openvla_image = (
    _libero_base
    .pip_install("transformers==4.40.1", "timm==0.9.10", "tokenizers==0.19.1",
                 "accelerate>=0.29", "sentencepiece", "protobuf")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(".", PROJ, **_LOCAL)
)

OPENVLA_MODEL = "openvla/openvla-7b-finetuned-libero-object"


def _apply_foldable_patch(vla, patch: str):
    """Milestone-0 CONVERSION PROBE: monkeypatch OpenVLA's non-foldable ops with FOLDABLE
    surrogates, NO finetune, to measure the per-op closed-loop fidelity drop before spending
    any training compute. Deployed-graph foldability is what we're testing the cost of.
      patch tokens (comma-joined): 'rmsnorm' (RMSNorm -> rational pade 1/sqrt), 'silu'
      (SiLU -> degree-3 poly fit). Returns #modules patched.
    NB: uses a per-forward scale s0=var.mean() (best-case centering) — an UPPER BOUND on
    fidelity; true foldable deployment freezes a calibrated per-layer s0."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    toks = set(patch.split(","))
    n = 0
    if "rmsnorm" in toks:
        # fit a [2/2] rational r(v) ~ v^{-1/2} on the log range [0.1,10]
        v = np.exp(np.linspace(np.log(0.1), np.log(10.0), 400)); t = v ** -0.5
        A = np.stack([np.ones_like(v), v, v * v, -t * v, -t * v * v], 1)
        c, *_ = np.linalg.lstsq(A, t, rcond=None)
        a_c = torch.tensor([c[0], c[1], c[2]], dtype=torch.float32)
        b_c = torch.tensor([1.0, c[3], c[4]], dtype=torch.float32)

        def make_rmsnorm_fwd(weight, eps, a_c, b_c):
            def fwd(x):
                xf = x.float()
                var = xf.pow(2).mean(-1, keepdim=True)
                s0 = var.mean().clamp_min(1e-12)
                vv = var / s0
                P = a_c[0] + a_c[1] * vv + a_c[2] * vv * vv
                Q = b_c[0] + b_c[1] * vv + b_c[2] * vv * vv
                inv = (P / Q) * torch.rsqrt(s0)                       # rational ~ 1/sqrt(var)
                return weight * (xf * inv).to(x.dtype)
            return fwd

        for m in vla.modules():
            if "RMSNorm" in type(m).__name__ and hasattr(m, "weight"):
                eps = getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))
                aa = a_c.to(m.weight.device); bb = b_c.to(m.weight.device)
                m.forward = make_rmsnorm_fwd(m.weight, eps, aa, bb)
                n += 1
    if "silu" in toks:
        # SiLU(x)=x*sigmoid(x); fit a degree-3 polynomial on [-8,8] (activation range)
        xs = np.linspace(-8, 8, 800); ys = xs / (1 + np.exp(-xs))
        pc = np.polyfit(xs, ys, 3)                                    # [c3,c2,c1,c0]
        pc_t = torch.tensor(pc[::-1].copy(), dtype=torch.float32)     # ascending

        def make_silu_fwd(pc_t):
            def fwd(x):
                xf = x.float()
                y = pc_t[0] + pc_t[1] * xf + pc_t[2] * xf * xf + pc_t[3] * xf * xf * xf
                return y.to(x.dtype)
            return fwd

        for m in vla.modules():
            if type(m).__name__ in ("SiLU", "SiLUActivation") or type(m).__name__.endswith("ACT2FN"):
                m.forward = make_silu_fwd(pc_t.to(next(vla.parameters()).device))
                n += 1
    return n


@app.function(image=openvla_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=6 * 3600)
def openvla_collect(n_eps: int = 2, max_task: int = 2, res: int = 256, save_res: int = 64,
                    horizon: int = 8, max_steps: int = 600, flip: str = "rot180",
                    num_steps_wait: int = 10, center_crop: bool = True,
                    use_init_states: bool = True, tag: str = "test", patch: str = "none"):
    """TEACHER integration + trajectory collection for distillation.
    Loads OpenVLA-7B (finetuned on LIBERO-Object), runs it IN our LIBERO sim, reports the
    teacher's success rate (sanity: ~90% => image/gripper convention correct; ~0% => wrong,
    try a different `flip`), and saves teacher trajectories (save_res image, instruction,
    executed 7-dim action, per-step) to /vol for distilling the tensor-pure student.
    Start SMALL (n_eps=2, max_task=2) to debug, then scale."""
    _bootstrap()
    import os, json, pickle
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    dev = "cuda"
    os.makedirs("/vol/hf", exist_ok=True)

    print("loading OpenVLA teacher (first run downloads ~14GB to /vol/hf)...")
    processor = AutoProcessor.from_pretrained(OPENVLA_MODEL, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        OPENVLA_MODEL, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(dev).eval()
    print("teacher loaded.")
    if patch != "none":
        np_ = _apply_foldable_patch(vla, patch)
        print(f"CONVERSION PROBE patch={patch}: monkeypatched {np_} modules (no finetune)")

    def prep_img(agent):
        if flip == "rot180":   a = agent[::-1, ::-1]
        elif flip == "vflip":  a = agent[::-1]
        elif flip == "hflip":  a = agent[:, ::-1]
        else:                  a = agent
        return np.ascontiguousarray(a)

    def _crop(a):
        # OpenVLA LIBERO finetunes were trained with center-crop aug; at eval crop central ~90%.
        if not center_crop: return a
        h, w = a.shape[:2]; m = int(round(0.05 * h))
        return a[m:h - m, m:w - m]

    @torch.no_grad()
    def teacher_action(agent_img, instr):
        # rot180, optional center crop, native res to the HF processor (which does the model's
        # own resize; double-resizing to 224 first degrades OpenVLA).
        img_in = Image.fromarray(_crop(prep_img(agent_img)))
        prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
        inputs = processor(prompt, img_in).to(dev, dtype=torch.bfloat16)
        act = vla.predict_action(**inputs, unnorm_key="libero_object", do_sample=False)
        return np.asarray(act, dtype=np.float32)          # (7,)

    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]          # no-op, gripper open (LIBERO settle)

    def fix_gripper(a):
        # OpenVLA gripper is [0,1] and sign-INVERTED vs the LIBERO env (official eval:
        # normalize_gripper_action [0,1]->[-1,1] + binarize, then invert_gripper_action).
        a = a.copy()
        g = 2.0 * float(a[-1]) - 1.0
        g = 1.0 if g >= 0 else -1.0
        a[-1] = -g
        return a

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    def build_state(obs):
        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = min(max_task, suite.n_tasks)
    traj, per_task, diag = [], {}, {}
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        init_states = None
        if use_init_states:
            init_states = _libero_init_states_or_raise(suite, ti)
        succ = 0
        for ep in range(n_eps):
            env.seed(ti * 100 + ep); obs = env.reset()
            if init_states is not None:                   # canonical eval init (official eval)
                obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait):               # let objects settle
                obs, _, _, _ = env.step(DUMMY)
            ep_frames, done_ok = [], False
            eef0 = build_state(obs)[:3].copy()
            for t in range(max_steps):
                agent = obs["agentview_image"]
                st = build_state(obs)
                a = teacher_action(agent, task.language)
                if ti == 0 and ep == 0 and t < 5:
                    diag.setdefault("first_actions", []).append(np.round(a, 3).tolist())
                if ti == 0 and ep == 0 and t == 50:
                    diag["eef_disp_50"] = np.round(build_state(obs)[:3] - eef0, 3).tolist()
                a_env = fix_gripper(a)                     # env-convention action (arm unchanged)
                img64 = np.asarray(Image.fromarray(prep_img(agent)).resize((save_res, save_res)),
                                   dtype=np.uint8)
                # (ti, ep, t, img64, instr, env-action(7), state(8)) — student learns env convention
                ep_frames.append((ti, ep, t, img64, task.language, a_env.copy(), st))
                obs, r, done, info = env.step(a_env.tolist())
                if r > 0: done_ok = True
                if done or done_ok: break
            succ += int(done_ok)
            # keep the trajectory regardless (successful ones are gold; failures still teach)
            for f in ep_frames:
                traj.append(f + (done_ok,))
        env.close()
        per_task[task.language] = round(succ / n_eps, 3)
        print(f"[teacher task {ti}] {task.language}: {succ}/{n_eps}")
        # INCREMENTAL SAVE after each task: a killed run leaves harvestable partial data
        # (openvla is slow + autoregressive, so full collections can exceed the wrapper's life).
        pickle.dump(traj, open(f"/vol/openvla_traj_{tag}.pkl", "wb"))
        with open(f"/vol/openvla_collect_{tag}.json", "w") as f:
            json.dump({"teacher_overall": round(float(np.mean(list(per_task.values()))), 3),
                       "per_task": per_task, "flip": flip, "center_crop": center_crop,
                       "use_init_states": use_init_states, "diag": diag,
                       "n_traj_steps": len(traj), "tasks_done": ti + 1, "partial": True}, f, indent=2)
        vol.commit()
    overall = round(float(np.mean(list(per_task.values()))), 3)
    out = f"/vol/openvla_traj_{tag}.pkl"
    pickle.dump(traj, open(out, "wb")); vol.commit()
    result = {"teacher_overall": overall, "per_task": per_task, "flip": flip,
              "center_crop": center_crop, "use_init_states": use_init_states,
              "diag": diag, "n_traj_steps": len(traj), "saved": out}
    with open(f"/vol/openvla_collect_{tag}.json", "w") as f: json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=4 * 3600)
def distill_student(traj_tag: str = "full", head: str = "product", steps: int = 15000,
                    horizon: int = 8, res: int = 64, eps_per_task: int = 20, max_task: int = 10,
                    max_steps: int = 280, num_steps_wait: int = 10,
                    exec_h: int = 8, n_factors: int = 4, curriculum: bool = True,
                    distill_teacher: bool = True, lambda_distill: float = 1.0,
                    success_only: bool = True, tag: str = "distill"):
    """Distill the tensor-pure student on OpenVLA teacher trajectories, then closed-loop rollout.
    Data from /vol/openvla_traj_{traj_tag}.pkl (openvla_collect). Attacks the data-scarcity /
    high-variance that caused seed s4=0.0: the teacher supplies many competent trajectories our
    ~14 demos/task did not."""
    _bootstrap()
    import os, json, pickle
    import numpy as np, torch
    from collections import defaultdict
    from PIL import Image
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig
    dev = "cuda"; H = horizon
    traj = pickle.load(open(f"/vol/openvla_traj_{traj_tag}.pkl", "rb"))
    if success_only:
        traj = [f for f in traj if f[7]]
    words = set()
    for f in traj: words.update(f[4].lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words): vocab[w] = len(vocab)
    T = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T] + [0] * max(0, T - len(ids)))[:T]
    eps = defaultdict(list)
    for f in traj: eps[(f[0], f[1])].append(f)
    samples = []
    for _, fs in eps.items():
        fs.sort(key=lambda z: z[2])
        A = np.stack([f[5] for f in fs])
        for i in range(len(fs) - H):
            samples.append((fs[i][3], fs[i][4], fs[i][6], A[i:i + H]))   # img, instr, state, chunk
    if not samples:
        return {"error": "no samples", "n_traj": len(traj)}
    d_a = int(np.asarray(samples[0][3]).shape[1])
    state_dim = int(np.asarray(samples[0][2]).shape[0])
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(s[1]) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head=head, n_factors=n_factors, distill_teacher=distill_teacher,
                    lambda_distill=lambda_distill, curriculum=curriculum)
    model = ChiVLA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=8e-4, max_steps=steps, warmup_frac=0.05)
    model.train()
    print(f"distilling head={head} on {len(samples)} teacher samples from {len(eps)} episodes")
    for step in range(steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
        idx = torch.randint(len(samples), (256,), device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs[idx], instr[idx], states[idx],
                            torch.zeros(256, dtype=torch.long, device=dev),
                            target_actions=actions[idx], progress=step / max(steps, 1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0: print(f"  step {step} loss {loss.item():.4f}")
    model.eval()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))
    @torch.no_grad()
    def predict_chunk(obs, instr_ids):
        raw = build_state(obs)
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(raw) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()
    n_tasks = min(max_task, suite.n_tasks); per_task = {}; canonical_tasks = 0
    close_sign = 1.0
    dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        instr_ids = torch.tensor([encode(task.language)], device=dev); succ = 0
        init_states = _libero_init_states_or_raise(suite, ti)
        if init_states is not None:
            canonical_tasks += 1
        for ep in range(eps_per_task):
            env.seed(ti * 100 + ep); obs = env.reset(); ok = False; t = 0
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step(dummy)
            while t < max_steps and not ok:
                chunk = predict_chunk(obs, instr_ids)
                for k in range(min(exec_h, H)):
                    a = chunk[k].copy(); a[-1] = close_sign if a[-1] > 0 else -close_sign
                    obs, r, done, info = env.step(a.tolist()); t += 1
                    if r > 0: ok = True
                    if done or ok or t >= max_steps: break
            succ += int(ok)
        env.close(); per_task[task.language] = round(succ / eps_per_task, 3)
        print(f"[distill {head} task {ti}] {task.language}: {succ}/{eps_per_task}")
    overall = round(float(np.mean(list(per_task.values()))), 3)
    result = {"overall": overall, "per_task": per_task, "head": head, "traj_tag": traj_tag,
              "n_samples": len(samples), "n_episodes": len(eps), "success_only": success_only,
              "steps": steps, "tasks_evaluated": n_tasks, "suite_tasks": suite.n_tasks,
              "eps_per_task": eps_per_task, "max_steps": max_steps,
              "num_steps_wait": num_steps_wait, "canonical_init_tasks": canonical_tasks,
              "canonical_init_states": canonical_tasks == n_tasks}
    with open(f"/vol/distill_{head}_{tag}.json", "w") as f: json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({"overall": overall, "n_samples": len(samples)}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def libero_rollout(steps: int = 6000, n_frames: int = 20000, horizon: int = 8, res: int = 64,
                   eps_per_task: int = 20, max_task: int = 10, max_steps: int = 280,
                   exec_h: int = 8):
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
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(raw) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()          # (H, d_a) un-normalized

    # FIXES (from agent synthesis): (A#1) binarize the mode-averaged bimodal gripper with a
    # hysteresis latch; (D#1) receding-horizon execution (re-predict every step) with ACT-style
    # temporal ensembling of the ARM dims. Neither touches the learned tensor network.
    close_sign = 1.0                                            # libero_diag: +1 closes gripper
    n_tasks = min(max_task, suite.n_tasks)
    sane, by_tau = None, {}
    for tau in [0.0, 0.1, 0.3]:                                 # gripper decision threshold sweep
        per_task = {}
        for ti in range(n_tasks):
            task = suite.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            instr_ids = torch.tensor([encode(task.language)], device=dev)
            succ = 0
            for ep in range(eps_per_task):
                env.seed(ti * 100 + ep); obs = env.reset()
                if sane is None:
                    sane = {"sim_state": np.round(build_state(obs), 3).tolist(),
                            "train_state_mean": np.round(s_mu, 3).tolist()}
                    print("STATE SANITY:", sane)
                buf, latched, ep_success = {}, False, False
                for t in range(max_steps):                      # re-predict every step
                    chunk = act(obs, instr_ids)                 # (H, d_a) raw
                    for k in range(H):
                        buf.setdefault(t + k, []).append(chunk[k])
                    preds = np.stack(buf.pop(t))                # predictions targeting step t
                    w = np.exp(-0.1 * np.arange(len(preds))[::-1]); w /= w.sum()
                    a = (w[:, None] * preds).sum(0)             # temporal-ensembled arm
                    g_raw = float(preds[-1, -1])                # newest raw gripper
                    if latched:
                        if g_raw < tau - 0.3: latched = False   # hysteresis
                    elif g_raw > tau:
                        latched = True
                    a[-1] = close_sign if latched else -close_sign
                    obs, r, done, info = env.step(a.tolist())
                    if r > 0:
                        ep_success = True
                    if done or ep_success:
                        break
                succ += int(ep_success)
            env.close()
            per_task[task.language] = round(succ / eps_per_task, 3)
            print(f"[tau={tau} task {ti}] {task.language}: {succ}/{eps_per_task}")
        by_tau[f"tau_{tau}"] = {"per_task": per_task,
                               "overall": round(float(np.mean(list(per_task.values()))), 3)}
        with open(f"{VOL_PATH}/libero_rollout.json", "w") as f:
            json.dump({"by_tau": by_tau, "state_sanity": sane, "eps_per_task": eps_per_task,
                       "fixes": "gripper_binarize+latch + receding-horizon temporal-ensemble(arm)"}, f, indent=2)
        vol.commit()
    print("RESULT:", json.dumps({k: v["overall"] for k, v in by_tau.items()}, indent=2))
    return {"by_tau": by_tau, "state_sanity": sane}


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=3 * 3600)
def build_frame_cache(n_frames: int = 100000, res: int = 64):
    """Stream + cache the LIBERO-Object frames to the volume (CPU, no GPU) so that
    multiple scaled training jobs can reuse the same cache without a concurrent
    first-download race. Idempotent: skips if the cache already exists."""
    _bootstrap()
    import os, pickle
    import numpy as np
    from datasets import load_dataset
    from PIL import Image
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    if os.path.exists(cache):
        frames = pickle.load(open(cache, "rb"))
        print(f"cache exists: {len(frames)} frames at {cache}")
        return {"cache": cache, "n_frames": len(frames), "existed": True}
    name = "lerobot/libero_object_image"
    ds = load_dataset(name, split="train", streaming=True); frames = []
    for ex in ds:
        img = ex["observation.images.image"]
        if not isinstance(img, Image.Image): img = Image.fromarray(np.array(img))
        frames.append((int(ex["episode_index"]), int(ex["frame_index"]),
                       np.asarray(img.resize((res, res)), dtype=np.uint8),
                       np.asarray(ex["observation.state"], dtype=np.float32),
                       np.asarray(ex["action"], dtype=np.float32), int(ex["task_index"])))
        if len(frames) >= n_frames: break
        if len(frames) % 10000 == 0: print(f"  streamed {len(frames)}")
    pickle.dump(frames, open(cache, "wb")); vol.commit()
    from collections import Counter
    tc = Counter(int(f[5]) for f in frames)
    print(f"cached {len(frames)} frames; tasks={dict(sorted(tc.items()))}")
    return {"cache": cache, "n_frames": len(frames), "tasks": dict(sorted(tc.items())), "existed": False}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def rollout_obs_check(n_frames: int = 20000, res: int = 64):
    """Decisive check for the 0% closed-loop: does the model see the SAME pixel
    distribution at rollout as in training? Compares a TRAINING frame (lerobot
    observation.images.image, resized) against the SIM agentview render under the
    rollout preprocessing (agentview_image[::-1] resized) AND without the flip, for
    the same task. Saves a side-by-side PNG and prints per-channel pixel stats — a
    visual/statistical smoking-gun for a camera/flip/resize mismatch."""
    _bootstrap()
    import os, pickle, json
    import numpy as np
    from PIL import Image
    from collections import defaultdict
    # training frames for task 0
    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    by_task = defaultdict(list)
    for f in frames:
        by_task[int(f[5])].append(f)
    stats = {}
    train_imgs = {}
    for ti in [0, 1]:
        fs = sorted(by_task[ti], key=lambda z: (z[0], z[1]))
        img = fs[0][2]                                   # first frame, uint8 (res,res,3)
        train_imgs[ti] = img
        stats[f"train_task{ti}"] = {"mean": np.round(img.mean(0).mean(0), 1).tolist(),
                                    "std": round(float(img.std()), 1)}
    # sim render for task 0/1
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    panels = []
    for ti in [0, 1]:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        env.seed(ti * 100); obs = env.reset()
        raw = np.ascontiguousarray(obs["agentview_image"])   # (res,res,3) uint8
        variants = {"vflip": raw[::-1], "hflip": raw[:, ::-1], "rot180": raw[::-1, ::-1]}
        cols = [train_imgs[ti]]
        for nm, arr in variants.items():
            v = np.asarray(Image.fromarray(np.ascontiguousarray(arr)).resize((res, res)))
            cols.append(np.full((res, 4, 3), 255, np.uint8)); cols.append(v)
        row = np.concatenate(cols, axis=1)
        env.close()
        panels.append(row)
    composite = np.concatenate([panels[0], np.full((4, panels[0].shape[1], 3), 255, np.uint8),
                                panels[1]], axis=0)
    Image.fromarray(composite).resize((composite.shape[1] * 3, composite.shape[0] * 3),
                                      Image.NEAREST).save(f"{VOL_PATH}/obs_check.png")
    vol.commit()
    print("OBS CHECK (columns: TRAIN | SIM_FLIP | SIM_NOFLIP; rows: task0, task1):")
    print(json.dumps(stats, indent=2))
    return stats


@app.function(image=image, gpu="A10G", timeout=1800)
def test_multimodal_heads(steps: int = 1500):
    """Validate the tensor-pure multimodal action heads (flow-matching + product-routing)
    BEFORE the expensive LIBERO run: (1) ChiVLA smoke for each head (forward+backward,
    shapes); (2) the collapse test — train each head vs the linear+MSE baseline on a
    synthetic CONDITIONAL-BIMODAL chunk and measure commit / mode-coverage. The linear
    head must collapse to the between-modes mean (commit≈0); the multimodal heads must
    commit to a real mode and cover both. Also checks BilinearFFN core foldability."""
    _bootstrap()
    import json
    import torch, torch.nn as nn, torch.nn.functional as F
    import numpy as np
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.flow_action import FlowMatchingActionHead
    from xvla.nn.product_routing import ProductRoutingHead
    dev = "cuda"; out = {}

    # (1) ChiVLA smoke: forward+backward for each head type
    for head in ["linear", "flow", "product", "quantile"]:
        cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=64, vit_layers=2, vit_heads=4,
                        vocab_size=16, max_instr_len=8, state_dim=8, n_embodiments=1,
                        dim=96, n_layers=2, n_heads=4, action_horizon=4, action_dim=7,
                        action_head=head, flow_steps=5, n_factors=3)
        m = ChiVLA(cfg).to(dev)
        B = 8
        img = torch.rand(B, 3, 32, 32, device=dev)
        instr = torch.randint(0, 16, (B, 8), device=dev)
        st = torch.randn(B, 8, device=dev)
        emb = torch.zeros(B, dtype=torch.long, device=dev)
        tgt = torch.randn(B, 4, 7, device=dev)
        a, loss = m(img, instr, st, emb, target_actions=tgt)
        loss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        out[f"vla_{head}"] = {"action_shape": list(a.shape), "loss": round(float(loss), 4),
                              "grad_flows": gnorm > 0, "params": m.num_params()}
        print(f"[vla {head}] action {list(a.shape)} loss {float(loss):.4f} grad {gnorm > 0}")

    # (2) collapse test on a synthetic conditional-bimodal chunk (heads in isolation)
    torch.manual_seed(0)
    dim, H, d_a = 32, 4, 7
    m_flat = H * d_a
    Wfeat = torch.randn(2, dim, device=dev)           # fixed context feature map
    base = torch.randn(2, H, d_a, device=dev) * 0.3   # per-context base chunk
    delta = torch.randn(H, d_a, device=dev)           # mode separation direction
    delta = delta / delta.norm() * 3.0

    def synth_batch(B):
        c = torch.randint(0, 2, (B,), device=dev)
        h = F.one_hot(c, 2).float() @ Wfeat            # (B, dim)
        p_plus = torch.where(c == 1, 0.7, 0.3)         # mode prob depends on context
        s = torch.where(torch.rand(B, device=dev) < p_plus, 1.0, -1.0)   # latent mode
        chunk = base[c] + s[:, None, None] * delta[None]                 # (B,H,d_a)
        chunk[:, :, -1] = s[:, None]                   # gripper dim = the mode sign
        chunk = chunk + 0.05 * torch.randn_like(chunk)
        return h, chunk, s

    res = {}
    # linear baseline
    lin = nn.Linear(dim, m_flat).to(dev)
    opt = torch.optim.Adam(lin.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = synth_batch(256)
        loss = F.mse_loss(lin(h), chunk.reshape(-1, m_flat))
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = synth_batch(2048)
    with torch.no_grad():
        pred = lin(h).reshape(-1, H, d_a)
    g = pred[:, :, -1].mean(1)                          # decoded gripper
    res["linear"] = {"gripper_commit_frac": round(float((g.abs() > 0.5).float().mean()), 3),
                     "gripper_abs_mean": round(float(g.abs().mean()), 3),
                     "gripper_sign_acc": round(float((g.sign() == s).float().mean()), 3)}

    # flow head
    flow = FlowMatchingActionHead(dim, d_a, H, depth=2, time_degree=3).to(dev)
    opt = torch.optim.Adam(flow.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = synth_batch(256)
        loss = flow.fm_loss(h, chunk)
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = synth_batch(2048)
    with torch.no_grad():
        samp = flow.sample(h, n_steps=20, n_samples=8)  # (B,8,H,d_a)
    g = samp[:, :, :, -1].mean(2)                        # (B,8) gripper per sample
    committed = samp[:, 0, :, -1].mean(1)               # first-sample gripper
    # coverage: across 8 samples for the ambiguous contexts, do we see both signs?
    gs = samp[:, :, :, -1].mean(2)                       # (B,8)
    both = ((gs > 0.3).any(1) & (gs < -0.3).any(1)).float().mean()
    res["flow"] = {"gripper_commit_frac": round(float((committed.abs() > 0.5).float().mean()), 3),
                   "gripper_abs_mean": round(float(committed.abs().mean()), 3),
                   "gripper_sign_acc": round(float((committed.sign() == s).float().mean()), 3),
                   "both_modes_covered_frac": round(float(both), 3),
                   "final_fm_loss": round(float(loss), 4)}

    # product head
    prod = ProductRoutingHead(dim, d_a, H, n_factors=3).to(dev)
    opt = torch.optim.Adam(prod.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = synth_batch(256)
        loss, _ = prod.loss(h, chunk)
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = synth_batch(2048)
    with torch.no_grad():
        dec = prod.decode(h)                             # (B,H,d_a)
        modes = prod.enumerate_modes(h)                  # (B,2^G,H,d_a)
    g = dec[:, :, -1].mean(1)
    gm = modes[:, :, :, -1].mean(2)                      # (B,2^G)
    both = ((gm > 0.3).any(1) & (gm < -0.3).any(1)).float().mean()
    res["product"] = {"gripper_commit_frac": round(float((g.abs() > 0.5).float().mean()), 3),
                      "gripper_abs_mean": round(float(g.abs().mean()), 3),
                      "gripper_sign_acc": round(float((g.sign() == s).float().mean()), 3),
                      "both_modes_available_frac": round(float(both), 3),
                      "final_loss": round(float(loss), 4)}

    # product head — Method 5 STC (straight-through Gumbel routing, temperature annealed)
    prod_st = ProductRoutingHead(dim, d_a, H, n_factors=3).to(dev)
    opt = torch.optim.Adam(prod_st.parameters(), 3e-3)
    for i in range(steps):
        h, chunk, s = synth_batch(256)
        prog = i / max(steps - 1, 1)
        st_temp = 1.5 * (0.3 / 1.5) ** prog
        loss, _ = prod_st.loss(h, chunk, straight_through=True, st_temp=st_temp, st_samples=4)
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = synth_batch(2048)
    with torch.no_grad():
        dec = prod_st.decode(h)
        modes = prod_st.enumerate_modes(h)
    g = dec[:, :, -1].mean(1)
    gm = modes[:, :, :, -1].mean(2)
    both = ((gm > 0.3).any(1) & (gm < -0.3).any(1)).float().mean()
    res["product_st"] = {"gripper_commit_frac": round(float((g.abs() > 0.5).float().mean()), 3),
                         "gripper_abs_mean": round(float(g.abs().mean()), 3),
                         "gripper_sign_acc": round(float((g.sign() == s).float().mean()), 3),
                         "both_modes_available_frac": round(float(both), 3),
                         "final_loss": round(float(loss), 4)}

    # (3) foldability: BilinearFFN core reproduces forward
    from xvla.nn.bilinear import BilinearFFN
    b = BilinearFFN(6, rank=8, out_dim=4, down_bias=False).to(dev).double()
    x = torch.randn(5, 6, device=dev).double()
    xb = torch.cat([torch.ones(5, 1, device=dev).double(), x], 1)
    T = b.dense_core()
    y_fold = torch.einsum("oij,bi,bj->bo", T, xb, xb)
    fold_err = float((b(x) - y_fold).abs().max())
    out["fold_err"] = fold_err
    out["collapse_test"] = res
    print("COLLAPSE TEST:", json.dumps(res, indent=2))
    print("FOLD ERR:", fold_err)
    return out


@app.function(image=image, gpu="A10G", timeout=900)
def rational_norm_check(nr_steps: int = 2):
    """Milestone-0 op check: does the FOLDABLE rational NR-rsqrt norm behave like RMSNorm
    (so it should preserve capability), and is it a fixed polynomial once s0 is frozen (so
    it folds)? Compares RationalNorm(nr_rsqrt) vs PerTokenRmsNorm over unit-scale AND
    tail-stressed inputs (the ρ≈2.8 regime), plus the attention head-dim variant."""
    _bootstrap()
    import json
    import torch
    from xvla.nn.normalization import RationalNorm, PerTokenRmsNorm
    dev = "cuda"; torch.manual_seed(0)
    rms = PerTokenRmsNorm().to(dev)
    rn = RationalNorm(variant="pade").to(dev)                # the fitted minimax-rational 1/√
    rn_nr = RationalNorm(variant="nr_rsqrt", nr_steps=nr_steps).to(dev)
    out = {"nr_steps": nr_steps, "variant": "pade"}
    # warm the running_ms on unit-scale activations (train mode), then freeze both
    for m in (rn, rn_nr):
        m.train()
        for _ in range(50):
            m(torch.randn(64, 384, device=dev))
        m.freeze(); m.eval()
    def reldiff(x, m=rn):
        a = rms(x); b = m(x)
        return float((a - b).norm() / (a.norm() + 1e-9))
    # unit scale + tail-stressed (×k) inputs — RMSNorm is scale-invariant; a good rational
    # approx should track it across the operating tail.
    out["reldiff_unit"] = round(reldiff(torch.randn(2048, 384, device=dev)), 4)
    out["pade_by_scale"] = {f"x{k}": round(reldiff(k * torch.randn(2048, 384, device=dev), rn), 4)
                            for k in [0.3, 0.5, 1.0, 1.5, 2.0, 2.8, 4.0]}
    out["nr_by_scale"] = {f"x{k}": round(reldiff(k * torch.randn(2048, 384, device=dev), rn_nr), 4)
                          for k in [0.5, 1.0, 2.0, 2.8]}
    # foldability sanity: with s0 frozen the map is a fixed function of the batch stats only
    # through ms(x); check determinism (same input → same output) and finiteness.
    xt = torch.randn(16, 384, device=dev)
    out["deterministic"] = bool(torch.allclose(rn(xt), rn(xt)))
    out["finite"] = bool(torch.isfinite(rn(xt)).all())
    # attention head-dim variant (d_h=32): NR rsqrt from s0=1 vs true rms over head_dim
    t = torch.randn(8, 8, 64, 32, device=dev)
    tru = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
    y = torch.ones(8, 8, 64, 1, device=dev)
    ms = t.pow(2).mean(-1, keepdim=True) + 1e-6
    for _ in range(3):
        y = y * (1.5 - 0.5 * ms * y * y)
    out["qk_rational_reldiff"] = round(float(((t * y) - tru).norm() / tru.norm()), 4)
    print("RATIONAL NORM CHECK:", json.dumps(out, indent=2))
    return out


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=12 * 3600)
def libero_rollout_head(head: str = "flow", steps: int = 6000, n_frames: int = 20000,
                        horizon: int = 8, res: int = 64, eps_per_task: int = 20,
                        max_task: int = 10, max_steps: int = 280, num_steps_wait: int = 10,
                        exec_h: int = 8, flow_steps: int = 10,
                        n_factors: int = 4, distill_teacher: bool = False,
                        lambda_recon: float = 0.0, lambda_distill: float = 0.0,
                        curriculum: bool = False, flow_decode: str = "mode",
                        flow_kwta_samples: int = 16, tag: str = "", norm: str = "per_token",
                        product_st: bool = False, seed: int = 0, qk_norm: str = "",
                        use_ema: bool = True, ema_decay: float = 0.999,
                        vision_encoder: str = "vit", conv_grid: int = 8,
                        load_ckpt: str = "", start_task: int = 0,
                        architecture: str = "chi", attn: str = "bilinear",
                        ffn: str = "bilinear", ffn_rank: int = 0,
                        vit_ffn_rank: int = 0):
    """CLOSED-LOOP eval with a TENSOR-PURE MULTIMODAL action head (flow-matching or
    product-routing) — the fix for the MSE mode-averaging that caused 0% closed-loop.
    Trains offline on LIBERO-Object then rolls out in MuJoCo. Unlike the linear-head
    rollout, the arm chunk is executed RECEDING-HORIZON WITHOUT cross-prediction
    temporal-ensembling (averaging samples from different modes would re-collapse them);
    the gripper is committed by the head itself (binarized by sign, an out-of-graph op).
    Also logs an OFFLINE commit diagnostic (gripper |g|>0.5 fraction, mode coverage)."""
    _bootstrap()
    import json, os, pickle, time
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig
    dev = "cuda"; H = horizon
    torch.manual_seed(seed); np.random.seed(seed)      # controlled init for seed-variance study
    if architecture == "conventional":
        if vision_encoder != "vit":
            raise ValueError("The controlled conventional twin currently requires vision_encoder='vit'.")
        # Per-block parameter matching. Bilinear attention has two more D->D maps than
        # softmax attention. Increasing SwiGLU rank from 3D to round((11D^2+7D)/(3D+2))
        # compensates that difference, giving rank 704 at D=192 and 1408 at D=384.
        attn, ffn = "softmax", "swiglu"
        norm, qk_norm = "per_token", "none"
        ffn_rank = ffn_rank or 1408
        vit_ffn_rank = vit_ffn_rank or 704
    elif architecture != "chi":
        raise ValueError("architecture must be 'chi' or 'conventional'")
    name = "lerobot/libero_object_image"

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
    print(f"{len(frames)} frames, head={head}")
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
    print(f"state_dim={state_dim} action_dim={d_a}")

    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head=head, flow_steps=flow_steps, n_factors=n_factors,
                    distill_teacher=distill_teacher, lambda_recon=lambda_recon,
                    lambda_distill=lambda_distill, curriculum=curriculum,
                    flow_decode=flow_decode, flow_kwta_samples=flow_kwta_samples,
                    norm=norm, qk_norm=(qk_norm or norm), product_st=product_st,
                    vision_encoder=vision_encoder, conv_grid=conv_grid,
                    attn=attn, ffn=ffn,
                    ffn_rank=(ffn_rank or None), vit_ffn_rank=(vit_ffn_rank or None))
    print(f"anchors: distill_teacher={distill_teacher} lambda_recon={lambda_recon} "
          f"lambda_distill={lambda_distill} curriculum={curriculum} "
          f"flow_decode={flow_decode}")
    model = ChiVLA(cfg).to(dev)
    params = model.num_params()
    print(f"architecture={architecture} attn={attn} ffn={ffn} "
          f"params={params / 1e6:.4f}M joint_ffn_rank={cfg.ffn_rank} "
          f"vit_ffn_rank={cfg.vit_ffn_rank}")
    ckpt_path = f"{VOL_PATH}/ckpt_{head}{tag}.pt"
    train_elapsed_s = None
    train_steps_per_s = None
    peak_gpu_memory_gb = None
    if load_ckpt:
        # CREDIT SAVER: skip the (expensive) training loop, load a saved model, go straight to
        # rollout. Data is still loaded above (cheap, cached) so vocab/normalization match.
        src = load_ckpt if load_ckpt.startswith("/") else f"{VOL_PATH}/{load_ckpt}"
        model.load_state_dict(torch.load(src, map_location=dev, weights_only=True))
        print(f"loaded checkpoint {src} — SKIPPING training")
        model.eval()
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05)
        tcfg = TrainConfig(train_bin="", val_bin="", lr=8e-4, max_steps=steps, warmup_frac=0.05)
        # weight EMA for eval — averages out late-training weight jitter, a cheap variance reducer
        ema = {n: p.detach().clone().float() for n, p in model.named_parameters()} if use_ema else None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        train_started = time.perf_counter()
        model.train()
        for step in range(steps + 1):
            for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
            idx = torch.randint(len(samples), (256,), device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(imgs[idx], instr[idx], states[idx],
                                torch.zeros(256, dtype=torch.long, device=dev),
                                target_actions=actions[idx], progress=step / max(steps, 1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ema is not None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema[n].mul_(ema_decay).add_(p.detach().float(), alpha=1.0 - ema_decay)
            if step % 1000 == 0: print(f"  step {step} loss {loss.item():.4f}")
        if ema is not None:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(ema[n].to(p.dtype))
        torch.cuda.synchronize()
        train_elapsed_s = time.perf_counter() - train_started
        train_steps_per_s = (steps + 1) / max(train_elapsed_s, 1e-9)
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        model.eval()
        # CHECKPOINT the trained model so re-evaluation never re-pays for training.
        torch.save(model.state_dict(), ckpt_path); vol.commit()
        print(f"saved checkpoint -> {ckpt_path}")

    # Matched batch-1 forward latency on cached observations. This includes the full vision,
    # language, state, joint-backbone, and linear action-head path, but not simulator stepping.
    with torch.no_grad():
        latency_idx = torch.tensor([0], device=dev)
        latency_emb = torch.zeros(1, dtype=torch.long, device=dev)
        for _ in range(20):
            model(imgs[latency_idx], instr[latency_idx], states[latency_idx], latency_emb)
        torch.cuda.synchronize()
        latency_started = time.perf_counter()
        for _ in range(100):
            model(imgs[latency_idx], instr[latency_idx], states[latency_idx], latency_emb)
        torch.cuda.synchronize()
        inference_latency_ms = 1000.0 * (time.perf_counter() - latency_started) / 100

    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)

    # ---- OFFLINE commit diagnostic (does the head commit the gripper vs collapse?) ----
    with torch.no_grad():
        idx = torch.randint(len(samples), (1024,), device=dev)
        pred, _ = model(imgs[idx], instr[idx], states[idx],
                        torch.zeros(1024, dtype=torch.long, device=dev))
        praw = pred * a_sd_t + a_mu_t                      # (B,H,d_a) raw
        graw = praw[:, :, -1].reshape(-1)                  # raw gripper commands
        tgt_g = (actions[idx] * a_sd_t + a_mu_t)[:, :, -1].reshape(-1)
        offline = {"gripper_commit_frac": round(float((graw.abs() > 0.5).float().mean()), 3),
                   "gripper_abs_mean": round(float(graw.abs().mean()), 3),
                   "target_gripper_abs_mean": round(float(tgt_g.abs().mean()), 3),
                   "arm_pred_std": round(float(praw[:, :, :-1].std()), 3)}
    print("OFFLINE COMMIT DIAG:", json.dumps(offline, indent=2))

    # ---- rollout in sim ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    @torch.no_grad()
    def predict_chunk(obs, instr_ids):
        raw = build_state(obs)
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(raw) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()      # (H, d_a) raw committed chunk

    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]     # arm still, gripper OPEN (settle)
    n_tasks = min(max_task, suite.n_tasks)
    sane, per_task, canonical_tasks = None, {}, 0
    for ti in range(start_task, n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        # Canonical LIBERO initial states plus a settle wait. A published comparison must also
        # match all ten tasks, the 280-step cap, and the number of trials. The OpenVLA paper
        # reports 88.4% for OpenVLA and 92.5% for Diffusion Policy.
        init_states = _libero_init_states_or_raise(suite, ti)
        if init_states is not None:
            canonical_tasks += 1
        succ = 0
        for ep in range(eps_per_task):
            env.seed(ti * 100 + ep); obs = env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step(DUMMY)
            if sane is None:
                sane = {"sim_state": np.round(build_state(obs), 3).tolist(),
                        "train_state_mean": np.round(s_mu, 3).tolist()}
                print("STATE SANITY:", sane)
            ep_success = False
            # RECEDING-HORIZON: predict a committed chunk, execute exec_h steps, re-predict.
            # NO cross-prediction averaging (would re-collapse multimodal samples).
            t = 0
            while t < max_steps and not ep_success:
                chunk = predict_chunk(obs, instr_ids)       # (H, d_a) raw
                for k in range(min(exec_h, H)):
                    a = chunk[k].copy()
                    a[-1] = close_sign if a[-1] > 0 else -close_sign   # commit gripper by sign
                    obs, r, done, info = env.step(a.tolist())
                    t += 1
                    if r > 0: ep_success = True
                    if done or ep_success or t >= max_steps: break
            succ += int(ep_success)
        env.close()
        per_task[task.language] = round(succ / eps_per_task, 3)
        print(f"[{head} task {ti}] {task.language}: {succ}/{eps_per_task}")
    overall = round(float(np.mean(list(per_task.values()))), 3)
    result = {"head": head, "overall": overall, "per_task": per_task,
              "offline_commit": offline, "state_sanity": sane, "steps": steps,
              "exec_h": exec_h, "flow_steps": flow_steps, "n_factors": n_factors,
              "flow_decode": flow_decode, "start_task": start_task,
              "architecture": architecture, "attention": attn, "ffn": ffn,
              "norm": norm, "qk_norm": (qk_norm or norm),
              "params": params, "params_M": round(params / 1e6, 4),
              "joint_ffn_rank": cfg.ffn_rank, "vit_ffn_rank": cfg.vit_ffn_rank,
              "train_elapsed_s": train_elapsed_s,
              "train_steps_per_s": train_steps_per_s,
              "peak_gpu_memory_gb": peak_gpu_memory_gb,
              "inference_latency_ms_batch1": inference_latency_ms,
              "task_indices": list(range(start_task, n_tasks)),
              "tasks_evaluated": len(per_task),
              "suite_tasks": suite.n_tasks, "eps_per_task": eps_per_task,
              "max_steps": max_steps, "num_steps_wait": num_steps_wait,
              "canonical_init_tasks": canonical_tasks,
              "canonical_init_states": canonical_tasks == len(per_task)}
    with open(f"{VOL_PATH}/libero_rollout_{head}{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({"head": head, "overall": overall, "offline": offline}, indent=2))
    return result


@app.function(image=libero_image, timeout=600)
def libero_scene_objects(max_task: int = 10):
    """Quick diagnostic: does a LIBERO-Object scene contain multiple distinct objects (so an
    instruction-swap causal intervention has something real to swap TO), or just the one named
    target? Lists every object body name present in each task's initial simulator state."""
    _bootstrap()
    import json, os
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    out = {}
    for ti in range(min(max_task, suite.n_tasks)):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        obs = env.reset()
        obj_names = sorted({k.rsplit("_", 1)[0] for k in obs.keys()
                            if k.endswith(("_pos", "_quat")) and not k.startswith("robot0")})
        out[ti] = {"language": task.language, "objects_in_scene": obj_names}
        env.close()
        print(f"[task {ti}] {task.language} -> objects: {obj_names}")
    with open(f"{VOL_PATH}/libero_scene_objects.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=600)
def libero_taskcov(n_frames: int = 20000, res: int = 64):
    """Is the training data actually spread across the eval tasks? The 20k frames are
    streamed in dataset order (task-blocked) → the tail tasks may have ZERO data →
    structurally 0% closed-loop. Histogram task_index over the cached frames."""
    import pickle, json
    from collections import Counter, defaultdict
    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    fc = Counter(int(f[5]) for f in frames)
    ep = defaultdict(set)
    for f in frames:
        ep[int(f[5])].add(int(f[0]))
    cov = {t: {"frames": fc[t], "episodes": len(ep[t])} for t in sorted(fc)}
    print("TASK COVERAGE:", json.dumps(cov, indent=2))
    print(f"distinct tasks with data: {len(cov)}  (eval uses task 0..7)")
    return cov


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def libero_diag(res: int = 64, n_frames: int = 20000):
    """Diagnostic for the 0% closed-loop: is the action/gripper convention right?
    (a) print raw lerobot action stats (esp. gripper dim encoding); (b) drive the sim
    with scripted actions and check the robot responds — which gripper sign closes, and
    whether a -z command lowers the end-effector. Isolates env/action convention from
    the model, no training/demos needed."""
    _bootstrap()
    import os, pickle, json
    import numpy as np
    # (a) lerobot raw action stats
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    astats = {}
    if os.path.exists(cache):
        frames = pickle.load(open(cache, "rb"))
        A = np.stack([f[4] for f in frames])                   # (N, d_a) raw actions
        astats = {"d_a": A.shape[1],
                  "per_dim_min": np.round(A.min(0), 3).tolist(),
                  "per_dim_max": np.round(A.max(0), 3).tolist(),
                  "per_dim_mean": np.round(A.mean(0), 3).tolist(),
                  "gripper_dim_unique_sample": np.round(np.unique(np.round(A[:2000, -1], 2))[:10], 2).tolist()}
        print("LEROBOT ACTION STATS:", json.dumps(astats, indent=2))
    # (b) scripted sim tests
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    env.seed(0); obs = env.reset()
    d_a = astats.get("d_a", 7)
    def zeros(): return [0.0] * d_a
    g0 = float(np.mean(np.abs(obs["robot0_gripper_qpos"])))
    a = zeros(); a[-1] = 1.0
    for _ in range(30): obs, *_ = env.step(a)
    g_plus = float(np.mean(np.abs(obs["robot0_gripper_qpos"])))
    obs = env.reset(); a = zeros(); a[-1] = -1.0
    for _ in range(30): obs, *_ = env.step(a)
    g_minus = float(np.mean(np.abs(obs["robot0_gripper_qpos"])))
    obs = env.reset(); z0 = float(obs["robot0_eef_pos"][2]); a = zeros(); a[2] = -1.0; a[-1] = -1.0
    for _ in range(20): obs, *_ = env.step(a)
    z1 = float(obs["robot0_eef_pos"][2])
    env.close()
    scripted = {"gripper_open_qpos_abs": round(g0, 4),
                "gripper_qpos_abs_after_+1": round(g_plus, 4),
                "gripper_qpos_abs_after_-1": round(g_minus, 4),
                "which_sign_closes": "+1" if g_plus < g_minus else "-1",
                "eef_z_start": round(z0, 4), "eef_z_after_-z_cmd": round(z1, 4),
                "z_went_down": z1 < z0}
    print("SCRIPTED SIM TEST:", json.dumps(scripted, indent=2))
    result = {"lerobot_action_stats": astats, "scripted_sim_test": scripted}
    with open(f"{VOL_PATH}/libero_diag.json", "w") as f: json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
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


def _libero_init_states_or_raise(suite, task_idx: int):
    """Load official LIBERO resets under PyTorch 2.6 and fail instead of substituting.

    LIBERO's packaged ``*.pruned_init`` files contain trusted NumPy arrays and predate the
    PyTorch 2.6 change that made ``torch.load(weights_only=True)`` the default.  The upstream
    loader does not pass ``weights_only=False``, so an unpatched call raises an UnpicklingError.
    Several older rollout paths caught that exception and quietly used random ``env.reset()``
    states, invalidating protocol comparisons.  This wrapper scopes the compatibility override
    to the official LIBERO loader, restores ``torch.load`` immediately afterward, and treats a
    missing or empty canonical state set as a hard error.
    """
    import torch

    original_torch_load = torch.load

    def trusted_libero_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_libero_load
    try:
        init_states = suite.get_task_init_states(task_idx)
    except Exception as exc:
        raise RuntimeError(
            f"Official LIBERO initial states failed to load for task {task_idx}. "
            "Random-reset fallback is forbidden."
        ) from exc
    finally:
        torch.load = original_torch_load

    if init_states is None or len(init_states) == 0:
        raise RuntimeError(
            f"Official LIBERO initial states are empty for task {task_idx}. "
            "Random-reset fallback is forbidden."
        )
    return init_states


@app.function(image=image, gpu="A10G", timeout=300)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def train_vla_libero_phase(steps: int = 4000, n_frames: int = 20000, horizon: int = 8,
                           res: int = 64, n_phases: int = 3):
    """PROTOTYPE (tensor-pure multimodal action via discrete-latent conditioning).

    Tests the thesis: p(a|obs) = Σ_z p(z|obs) p(a|obs,z). z = a PHASE latent labelled
    FOR FREE from the demo gripper signal (reach/grasp/lift). We (a) measure, on the
    data alone, how much conditioning on z unimodalizes the action distribution
    (esp. the bimodal gripper dim); (b) train a phase-CONDITIONED χ-VLA (z enters as
    one extra token, action head stays linear+MSE, a LINEAR phase head emits p(z|obs)
    logits) vs a phase-BLIND baseline; (c) report action MSE (blind vs conditioned on
    true z vs conditioned on OUT-OF-GRAPH argmax ẑ) + gripper-dim MSE + phase-head acc.
    Everything nonlinear (argmax over z) is out-of-graph; the exported graph stays
    linear/foldable. Requires the frame cache from train_vla_libero."""
    _bootstrap()
    import json, os, pickle
    from collections import defaultdict
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig
    from xvla.train.phase import (label_chunks_sign, reach_grasp_lift_phase,
                                  conditional_vs_marginal_spread)

    name = "lerobot/libero_object_image"
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
                for _, r in df.iterrows():
                    tasks[int(r[icol])] = str(r[tcol])
            else:
                for line in open(tp):
                    r = json.loads(line); tasks[int(r["task_index"])] = r["task"]
            if tasks:
                break
        except Exception as e:
            print(f"  {tf} failed: {e}")

    words = set()
    for t in tasks.values():
        words.update(t.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words):
        vocab[w] = len(vocab)
    T = 32

    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T] + [0] * max(0, T - len(ids)))[:T]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    assert os.path.exists(cache), f"run train_vla_libero first to build {cache}"
    with open(cache, "rb") as f:
        frames = pickle.load(f)
    print(f"{len(frames)} frames")

    # Group by episode; per episode compute the reach/grasp/lift phase timeline from
    # the gripper action dim (index -1), then build (img, instr, state, chunk, phase).
    H = horizon
    eps = defaultdict(list)
    for fr in frames:
        eps[fr[0]].append(fr)
    samples, phases = [], []
    for ep, fs in eps.items():
        fs.sort(key=lambda z: z[1])
        grip_seq = np.array([fr[4][-1] for fr in fs], dtype=np.float32)   # gripper timeline
        if n_phases == 3:
            ph_seq = reach_grasp_lift_phase(grip_seq, close_positive=True)
        for i in range(len(fs) - H):
            acts = np.stack([fs[i + k][4] for k in range(H)])
            samples.append((fs[i][2], fs[i][5], fs[i][3], acts))
            phases.append(int(ph_seq[i]) if n_phases == 3 else None)
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    if n_phases == 2:
        lab_np = label_chunks_sign(A, gripper_dim=-1, close_positive=True)
    else:
        lab_np = np.array(phases, dtype=np.int64)
    d_a = A.shape[-1]; state_dim = S.shape[-1]

    # (a) DATA-ONLY unimodalization diagnostic (no model).
    spread = conditional_vs_marginal_spread(A, lab_np, gripper_dim=-1)
    print("UNIMODALIZATION (data-only):", json.dumps(
        {k: spread[k] for k in ("gripper_marginal_std", "gripper_within_phase_std", "gripper_ratio")}, indent=2))
    print("  per-phase counts:", {int(z): int((lab_np == z).sum()) for z in np.unique(lab_np)})

    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    dev = "cuda"
    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    labels = torch.tensor(lab_np, device=dev)
    grip_idx = d_a - 1
    n = len(samples); perm = torch.randperm(n, device=dev); ntr = int(n * 0.9)
    tr, te = perm[:ntr], perm[ntr:]

    def make_cfg(np_):
        return VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                         vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim,
                         n_embodiments=1, dim=384, n_layers=8, n_heads=12,
                         action_horizon=H, action_dim=d_a, n_phases=np_)

    def train(model, conditioned):
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05)
        tcfg = TrainConfig(train_bin="", val_bin="", lr=8e-4, max_steps=steps, warmup_frac=0.05)
        model.train()
        for step in range(steps + 1):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, tcfg)
            idx = tr[torch.randint(len(tr), (256,), device=dev)]
            opt.zero_grad(set_to_none=True)
            e = torch.zeros(len(idx), dtype=torch.long, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if conditioned:
                    _, loss, _ = model(imgs[idx], instr[idx], states[idx], e,
                                       target_actions=actions[idx], phase_id=labels[idx],
                                       phase_labels=labels[idx], phase_weight=1.0, return_phase=True)
                else:
                    _, loss = model(imgs[idx], instr[idx], states[idx], e, target_actions=actions[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 500 == 0:
                print(f"  [{'cond' if conditioned else 'blind'}] step {step} loss {loss.item():.5f}")

    @torch.no_grad()
    def eval_mse(model, mode):
        """mode: blind | true_phase | pred_phase (out-of-graph argmax over p(z|obs))."""
        model.eval(); tot = cnt = gtot = gcnt = 0.0; pcorr = ptot = 0
        for i in range(0, len(te), 512):
            b = te[i:i + 512]; e = torch.zeros(len(b), dtype=torch.long, device=dev)
            if mode == "blind":
                a, _ = model(imgs[b], instr[b], states[b], e)
            else:
                # one pass to read LINEAR phase logits p(z|obs)
                _, _, logits = model(imgs[b], instr[b], states[b], e, return_phase=True)
                zhat = logits.argmax(-1)                       # OUT-OF-GRAPH argmax
                pcorr += int((zhat == labels[b]).sum()); ptot += len(b)
                z = labels[b] if mode == "true_phase" else zhat
                a, _, _ = model(imgs[b], instr[b], states[b], e, phase_id=z, return_phase=True)
            d = (a - actions[b]).pow(2)
            tot += d.sum().item(); cnt += a.numel()
            gtot += d[:, :, grip_idx].sum().item(); gcnt += d[:, :, grip_idx].numel()
        acc = (pcorr / ptot) if ptot else None
        return {"mse": tot / cnt, "gripper_mse": gtot / gcnt, "phase_acc": acc}

    torch.manual_seed(0)
    blind = ChiVLA(make_cfg(0)).to(dev)
    print(f"blind params={blind.num_params()/1e6:.2f}M ; conditioned n_phases={n_phases}")
    train(blind, conditioned=False)
    r_blind = eval_mse(blind, "blind")

    torch.manual_seed(0)
    cond = ChiVLA(make_cfg(n_phases)).to(dev)
    train(cond, conditioned=True)
    r_true = eval_mse(cond, "true_phase")
    r_pred = eval_mse(cond, "pred_phase")

    result = {
        "n_samples": n, "n_phases": n_phases, "d_a": d_a,
        "data_unimodalization": {k: spread[k] for k in
            ("gripper_marginal_std", "gripper_within_phase_std", "gripper_ratio",
             "unimodalization_ratio")},
        "blind":         {k: round(v, 5) if isinstance(v, float) else v for k, v in r_blind.items()},
        "cond_true_z":   {k: round(v, 5) if isinstance(v, float) else v for k, v in r_true.items()},
        "cond_pred_z":   {k: round(v, 5) if isinstance(v, float) else v for k, v in r_pred.items()},
    }
    with open(f"{VOL_PATH}/vla_libero_phase.json", "w") as f:
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


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def extended_abstract_local_replay_corrected(seed: int = 0, tag: str = "v2"):
    """Run corrected rational-aware local coefficient replay on one frozen checkpoint."""
    import json
    import os
    import subprocess

    checkpoints = {
        0: "ckpt_linear_rat_vit_s0_v2.pt",
        1: "ckpt_linear_rat_vit_s1.pt",
        2: "ckpt_linear_rat_vit_s2.pt",
    }
    if seed not in checkpoints:
        raise ValueError("seed must be 0, 1, or 2")
    result_name = f"extended_abstract_local_replay_{tag}_s{seed}.json"
    result_path = f"{VOL_PATH}/{result_name}"
    if os.path.exists(result_path):
        raise FileExistsError(result_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{PROJ}:/opt/LIBERO"
    command = [
        sys.executable,
        f"{PROJ}/athena/run_extended_abstract_local_replay_v1.py",
        "--mode", "full",
        "--seed", str(seed),
        "--checkpoint", f"{VOL_PATH}/{checkpoints[seed]}",
        "--cache", f"{VOL_PATH}/libero_frames_100000_64.pkl",
        "--result", result_path,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJ,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"corrected local replay exited {completed.returncode}; result={result_path}"
        )
    vol.commit()
    with open(result_path) as stream:
        return json.load(stream)


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=6 * 3600)
def extended_abstract_route_semantics(
    seed: int = 0,
    task_start: int = 4,
    task_end: int = 5,
    smoke: bool = True,
    tag: str = "v1",
):
    """Measure named input-path sensitivity along the block-0 weight subspace."""
    import hashlib
    import json
    import os
    import subprocess

    checkpoints = {
        0: "ckpt_linear_rat_vit_s0_v2.pt",
        1: "ckpt_linear_rat_vit_s1.pt",
        2: "ckpt_linear_rat_vit_s2.pt",
    }
    if seed not in checkpoints:
        raise ValueError("seed must be 0, 1, or 2")
    if smoke:
        if (seed, task_start, task_end) != (0, 4, 5):
            raise ValueError("strict smoke is fixed to seed 0 and official task 4")
        result_name = f"extended_abstract_route_semantics_{tag}_smoke.json"
    else:
        if not (4 <= task_start < task_end <= 10):
            raise ValueError("full shards must use held-out official tasks 4 through 9")
        result_name = (
            f"extended_abstract_route_semantics_{tag}_s{seed}_t{task_start}_{task_end}.json"
        )
    result_path = f"{VOL_PATH}/{result_name}"
    if os.path.exists(result_path):
        raise FileExistsError(result_path)

    runner_path = f"{PROJ}/athena/extended_abstract_route_semantics_v1.py"
    with open(runner_path, "rb") as stream:
        runner_sha256 = hashlib.sha256(stream.read()).hexdigest()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{PROJ}:/opt/LIBERO"
    environment["XVLA_ROUTE_SEMANTICS_V1_SHA256"] = runner_sha256
    command = [
        sys.executable,
        runner_path,
        "--mode", "strict_smoke" if smoke else "full",
        "--checkpoint", f"{VOL_PATH}/{checkpoints[seed]}",
        "--checkpoint-seed", str(seed),
        "--cache", f"{VOL_PATH}/libero_frames_100000_64.pkl",
        "--output", result_path,
        "--task-start", str(task_start),
        "--task-end", str(task_end),
        "--discovery-episodes-per-task", "1" if smoke else "10",
        "--evaluation-episodes-per-task", "1" if smoke else "20",
        "--random-controls", "1" if smoke else "4",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJ,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"route semantics exited {completed.returncode}; result={result_path}"
        )
    vol.commit()
    with open(result_path) as stream:
        return json.load(stream)


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
    import hashlib
    import importlib.metadata
    import json
    import platform
    import random
    from pathlib import Path

    import numpy as np
    import torch
    from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
    from xvla.train.odt import (export_cores, unroll_forward, downstream_gram,
                                local_gram, truncation_curve, top_projector, accuracy)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tl, vl = _svhn_loaders(flatten=True)
    vol.commit()
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

    checkpoint_path = Path(VOL_PATH) / "odt_experiment_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    dataset_files = {}
    for path in sorted((Path(VOL_PATH) / "svhn").glob("*.mat")):
        dataset_files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    source_paths = [
        "modal_app.py",
        "xvla/__init__.py",
        "xvla/models/__init__.py",
        "xvla/models/lm.py",
        "xvla/models/chi_mlp.py",
        "xvla/nn/__init__.py",
        "xvla/nn/attention.py",
        "xvla/nn/baselines.py",
        "xvla/nn/bilinear.py",
        "xvla/nn/block.py",
        "xvla/nn/homogeneous.py",
        "xvla/nn/normalization.py",
        "xvla/train/__init__.py",
        "xvla/train/odt.py",
    ]
    source_sha256 = {
        relative: digest(Path(PROJ) / relative)
        for relative in source_paths
    }

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
        "schema": "xvla-exact-mlp-odt-v2",
        "base_acc": base_acc, "unroll_acc": acc_unroll,
        "reconstruction_max_abs": recon,
        "global_curve": cg, "local_curve": cl,
        "removable_dims_1pct": removable, "removable_frac": removable / 32,
        "spectrum": [round(float(s), 6) for s in spec.tolist()],
        "global_beats_local_at": [k for k in ks if cg[k] > cl[k] + 1e-4],
        "protocol": {
            "epochs": epochs,
            "batch_size": 256,
            "seed": 0,
            "analysis_examples": 4000,
            "bond": bond,
            "rank_grid": ks,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 2e-3,
                "weight_decay": 0.05,
                "betas": [0.9, 0.95],
                "gradient_clip_norm": 1.0,
            },
            "model": {
                "input_dimension": 3072,
                "width": 32,
                "layers": 3,
                "classes": 10,
                "normalization": "scalar_rbn",
            },
            "svhn_normalization_mean": [0.4377, 0.4438, 0.4728],
            "svhn_normalization_std": [0.198, 0.201, 0.197],
        },
        "dataset_files": dataset_files,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": digest(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "source_sha256": source_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "claim_boundary": (
            "This is exact weight-only ODT for a three-layer tree-structured chi MLP. "
            "It does not establish exact global ODT for the residual-attention policy."
        ),
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


def _stl_loaders(batch=128, root="/vol/stl", res=64):
    import torch
    import torchvision as tv
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(res), T.ToTensor(),
                    T.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))])
    tr = tv.datasets.STL10(root, split="train", download=True, transform=tf)
    te = tv.datasets.STL10(root, split="test", download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, num_workers=4, drop_last=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False, num_workers=4))


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def g7_basis_overlap(epochs: int = 12):
    """G7: does the DATA-FREE weight-only ODT basis coincide with the DATA-DRIVEN
    output-sensitivity Gram basis (the fallback used through attention)? On the exact
    conv-spatial classifier, compute both input-space bases and their top-k subspace
    overlap. High overlap ⇒ the data-driven method (all we can compute through attention)
    is a faithful proxy for the weight-only one — licensing the 'weight-derived' story
    past feed-forward bonds."""
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_conv import ShallowBilinear
    from xvla.train.topology import class_quadratics, input_gram
    tl, vl = _svhn_loaders(flatten=False)
    vol.commit()
    torch.manual_seed(0)
    model = ShallowBilinear(mode="conv", readout="spatial", grid=8, width=48, kernel=5,
                            in_ch=3, hw=32, num_classes=10)
    acc = _train_shallow(model, tl, vl, epochs, lr=2e-3)
    model = model.cuda().double().eval()
    D = 3 * 32 * 32
    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 2000:
            break
    X = torch.cat(xs)[:2000].cuda().double(); Y = torch.cat(ys)[:2000].cuda()
    # data-free weight-only basis: eigvecs of Σ_c Q_c²
    Q, _, _ = class_quadratics(model, "cuda")
    Gwf = torch.zeros(D, D, device="cuda", dtype=torch.float64)
    for c in range(Q.shape[0]):
        Gwf += Q[c] @ Q[c]
    Vw = torch.linalg.eigh(Gwf)[1].flip(1); del Q, Gwf; torch.cuda.empty_cache()
    # data-driven basis: input output-sensitivity Gram
    Gdd = input_gram(lambda z: model.logits(z), X, Y, "cuda")
    Vd = torch.linalg.eigh(Gdd)[1].flip(1)
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    overlap = {int(k): round((Vw[:, :k].T @ Vd[:, :k]).pow(2).sum().item() / k, 3) for k in ks}
    rand_base = {int(k): round(k / D, 3) for k in ks}
    result = {"clf_acc": round(acc, 4), "subspace_overlap": overlap,
              "random_baseline": rand_base,
              "note": "overlap in [0,1]; weight-only (data-free) vs data-driven Gram basis, top-k"}
    with open(f"{VOL_PATH}/g7_basis_overlap.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def topology_scale_stl(epochs: int = 20, seeds: int = 2):
    """G8: does topology→coherence hold BEYOND 32×32? STL-10 at 64×64 (D=12288),
    dense (ChiMLP) vs conv (ConvBilinearDeep), depth-3, via the data-driven input-space
    Gram (exact-Q is memory-bound at this resolution). Reports acc + atom locality ratio
    + global-vs-random faithfulness, mean±std over seeds."""
    _bootstrap()
    import json
    import torch
    from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
    from xvla.models.chi_conv import ConvBilinearDeep
    from xvla.train.topology import input_analysis
    tl, vl = _stl_loaders(res=64)
    vol.commit()
    D = 3 * 64 * 64
    xs, ys = [], []
    for imgs, labels in vl:
        xs.append(imgs); ys.append(labels)
        if sum(t.shape[0] for t in xs) >= 1500:
            break
    X = torch.cat(xs)[:1500].cuda(); Y = torch.cat(ys)[:1500].cuda()
    agg = {}
    for arch in ["dense", "conv"]:
        accs, ratios, g64, r64 = [], [], [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            if arch == "dense":
                model = ChiMLP(ChiMLPConfig(in_dim=D, dim=48, n_layers=3, num_classes=10,
                                            norm="scalar_rbn"))
                f = lambda z: model(z)[0]
            else:
                model = ConvBilinearDeep(depth=3, width=48, kernel=5, grid=8, in_ch=3, hw=64,
                                         num_classes=10)
                f = lambda z: model.logits(z)
            acc = _train_shallow(model, tl, vl, epochs, lr=2e-3)
            model.cuda().eval()
            res = input_analysis(f, X, Y, "cuda", ks=(1, 2, 4, 8, 16, 32, 64, 128, 256))
            accs.append(acc); ratios.append(res["locality_ratio"])
            g64.append(res["global_curve"][64]); r64.append(res["random_curve"][64])
            torch.cuda.empty_cache()
        def ms(v):
            t = torch.tensor(v); return [round(t.mean().item(), 3), round(t.std().item(), 3)]
        agg[arch] = {"acc": ms(accs), "locality_ratio": ms(ratios),
                     "global@64": ms(g64), "random@64": ms(r64)}
        print(f"[STL64 {arch}] acc {ms(accs)} locality {ms(ratios)} g@64 {ms(g64)} r@64 {ms(r64)}")
    with open(f"{VOL_PATH}/topology_scale_stl.json", "w") as f:
        json.dump(agg, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(agg, indent=2))
    return agg


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
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


@app.function(
    image=libero_image,
    gpu="A10G",
    cpu=12,
    memory=32768,
    volumes={VOL_PATH: vol},
    timeout=2 * 3600,
)
def odt_libero_action(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                      res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                      norm: str = "rational", n_eval: int = 2048):
    """Interpretability PAYOFF on the ACTUAL trained LIBERO policy (not a component, not the
    synthetic VLA) — promotes Exp B/C's causal-low-rank result to the real, closed-loop-
    competitive checkpoint. Loads a saved libero_rollout_head checkpoint (default: the
    fully-foldable rational-norm ViT model, DEVLOG cont.51-52), builds the data-driven
    output-sensitivity Gram at the VISUAL-PATCH bond (downstream = the full causal bilinear-
    attention backbone + linear head) per action GROUP {eef-translation, rotation, gripper},
    then measures OFFLINE action faithfulness: truncating the bond onto its top-k GLOBAL
    eigendirections vs a random k-subspace. Honesty: this is a data-driven Gram (norm-agnostic,
    transfers verbatim to any checkpoint via `ckpt`), NOT exact weight-only ODT — the
    causal/softmax-free-attention backbone is Level-C/open (odt.py); exact fold only applies to
    feedforward chains. No training, no sim — pure forward/backward on cached frames."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.odt import random_projector

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    D = cfg.dim
    print(f"loaded {ckpt}: D={D} d_a={d_a} n_samples={len(samples)}")

    # ---- held-out eval batch ----
    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states_raw = np.stack([samples[i][2] for i in idx])
    states = torch.tensor((states_raw - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions_raw = np.stack([samples[i][3] for i in idx])
    Atrue = torch.tensor((actions_raw - a_mu) / a_sd, dtype=torch.float32, device=dev)   # (B,H,d_a)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    def build_seq(vis, sl):
        B = vis.shape[0]
        bos = model.bos.expand(B, -1, -1)
        instr_e = model.tok_emb(instr[sl])
        st = model.state_proj(states[sl])[:, None]
        embe = model.embodiment_emb(emb0[sl])[:, None]
        aq = model.action_queries.expand(B, -1, -1)
        x = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
        return x + model.pos_emb[:, :x.shape[1]]

    @torch.no_grad()
    def vis_tokens_fixed():
        return model._visual_tokens(imgs)                          # (B, Nv, D) — the bond

    def from_vis(vis_in, sl, P=None):
        v = vis_in @ P.T if P is not None else vis_in               # project each patch (B,Nv,D)
        x = build_seq(v, sl)
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        h = model.backbone(x, mask=mask)
        h = model.norm_out(h)
        return model.action_head(h[:, -H:])                        # (B,H,d_a)

    groups = {"eef_transl": list(range(min(3, d_a))), "rotation": list(range(3, min(6, d_a))),
              "gripper": [d_a - 1]}

    # ---- action-GROUP-conditioned Grams at the visual-patch bond ----
    vis_fixed = vis_tokens_fixed()
    grams = {}
    for gname, dims in groups.items():
        if not dims: continue
        G = torch.zeros(D, D, device=dev, dtype=torch.float64)
        n = 0
        for i in range(0, vis_fixed.shape[0], 256):
            sl = slice(i, i + 256)
            vb = vis_fixed[sl].detach().requires_grad_(True)
            act = from_vis(vb, sl)
            sel = act[:, :, dims].sum()
            gsel, = torch.autograd.grad(sel, vb)
            g = gsel.reshape(-1, D).double()
            G += g.T @ g; n += g.shape[0]
        grams[gname] = G / n
    print("built Grams for groups:", list(grams.keys()))

    def topk_proj(G, k):
        ev, V = torch.linalg.eigh(G)
        return V.flip(1)[:, :k]

    ks = [4, 8, 16, 32, 64]
    rng = torch.Generator(device=dev).manual_seed(0)

    @torch.no_grad()
    def group_mse(P, dims):
        tot = n = 0.0
        for i in range(0, vis_fixed.shape[0], 256):
            sl = slice(i, i + 256)
            pred = from_vis(vis_fixed[sl], sl, P)
            tot += (pred[:, :, dims] - Atrue[sl][:, :, dims]).pow(2).sum().item()
            n += pred[:, :, dims].numel()
        return round(tot / n, 5)

    faith = {}
    full_mse = {}
    for gname, dims in groups.items():
        if not dims or gname not in grams: continue
        full_mse[gname] = group_mse(None, dims)
        Vg = topk_proj(grams[gname], max(ks))
        curve_g, curve_r = {}, {}
        for k in ks:
            Pg = (Vg[:, :k] @ Vg[:, :k].T).float()
            curve_g[k] = group_mse(Pg, dims)
            curve_r[k] = group_mse(random_projector(D, k, dev, rng).float(), dims)
        faith[gname] = {"full_D": D, "full_mse": full_mse[gname], "global": curve_g, "random": curve_r}

    # ---- SPATIAL COHERENCE of the top causal direction per group (ties to the topology
    # thesis: does the visual-patch mechanism concentrate on a few patches, or stay diffuse?
    # This is the first time this is measured on a real, deployed, closed-loop policy rather
    # than an SVHN/CIFAR classifier). Nv patches laid out on a sqrt(Nv) x sqrt(Nv) grid.
    Nv = vis_fixed.shape[1]
    grid = int(round(Nv ** 0.5))
    frac = 0.1
    k_top = max(1, int(round(frac * Nv)))
    rng_loc = torch.Generator(device=dev).manual_seed(0)

    @torch.no_grad()
    def patch_locality(v):
        # v: (D,) direction. Score each patch position by |vis . v|^2, averaged over the batch.
        proj = torch.einsum("bnd,d->bn", vis_fixed.double(), v.double())    # (B, Nv)
        energy = (proj ** 2).mean(0)                                        # (Nv,)
        return (torch.topk(energy, k_top).values.sum() / energy.sum().clamp_min(1e-30)).item()

    coherence = {}
    for gname in grams:
        v_top = topk_proj(grams[gname], 1)[:, 0]
        rand_locs = [patch_locality(torch.randn(D, generator=rng_loc, device=dev)) for _ in range(30)]
        coherence[gname] = {"top_causal_dir_locality": round(patch_locality(v_top), 4),
                            "random_dir_locality_mean": round(sum(rand_locs) / len(rand_locs), 4),
                            "grid": grid, "frac": frac}
    print("COHERENCE:", json.dumps(coherence, indent=2))

    result = {"ckpt": ckpt, "norm": norm, "vision_encoder": vision_encoder, "D": D, "d_a": d_a,
              "n_eval": len(idx), "faithfulness_by_group": faith, "coherence_by_group": coherence}
    with open(f"{VOL_PATH}/odt_libero_action_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def odt_norm_bond_diagnostic(vit_ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt",
                             conv_ckpt: str = "ckpt_linear_rat_conv_s0.pt",
                             n_frames: int = 100000, res: int = 64, horizon: int = 8,
                             norm: str = "rational", n_eval: int = 1024, group: str = "gripper"):
    """Diagnose the conv+rational coherence-below-random surprise (DEVLOG cont.59): a
    code-verified architectural asymmetry between the two vision encoders' `.features()` --
    ChiConvEncoder applies a final per-token norm before returning tokens (xvla/models/vit.py
    ChiConvEncoder.features), ChiViT does not (norm_out is dead code in ChiViT.features's path,
    only used in its separate classifier .forward()). Since the deployed policy's coherence
    metric (odt_libero_action's patch_locality) measures CROSS-PATCH magnitude concentration,
    and a per-token norm equalizes every patch's overall magnitude by construction, conv's extra
    norm call may be erasing exactly the signal the metric is designed to detect -- a metric/bond
    mismatch, not necessarily evidence against real conv locality. Tests this by Gramming BOTH
    encoders at BOTH bonds (native = the real deployed bond, unchanged; alternate = the other
    encoder's convention) on their ALREADY-TRAINED checkpoints (zero retraining) -- for the
    alternate bond, the norm is applied or skipped before the unchanged downstream computation.
    The alternate paths therefore diagnose sensitivity to normalization placement and are not
    deployed-policy or in-distribution results. The native paths remain byte-for-byte unchanged."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)
    groups = {"eef_transl": list(range(min(3, d_a))), "rotation": list(range(3, min(6, d_a))),
              "gripper": [d_a - 1]}
    dims = groups[group]

    def load_model(ckpt, vision_encoder):
        cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                        vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                        dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                        action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
        m = ChiVLA(cfg).to(dev)
        m.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
        m.eval()
        return m, cfg

    def get_leaf(model, cfg, bond_variant):
        """Returns the (B, Nv, cfg.dim) leaf tensor the Gram is built w.r.t. -- POST
        `model.vis_proj`, exactly the bond `ChiVLA._visual_tokens`/`odt_libero_action` use, so
        'native' results here are directly comparable to the already-reported 1.08-1.41x (vit) /
        0.81-0.87x (conv) numbers. The norm-variant choice happens BEFORE vis_proj (matching
        the real architecture: norm is per-encoder, vis_proj is the shared joint-space lift), so
        changing this choice also changes the distribution presented to vis_proj and the
        backbone. The alternate result is therefore a normalization-placement diagnostic rather
        than a clean intervention at an otherwise unchanged bond.
        'native' = the real deployed pre-vis_proj tensor, unchanged. 'alt' = the other encoder's
        norm-placement convention (vit: apply its otherwise-dead-code norm_out; conv: skip its
        trained-in final norm)."""
        with torch.no_grad():
            if cfg.vision_encoder == "vit":
                raw_native = model.vision.features(imgs)                       # (B, Nv, vit_dim), no norm
                raw = raw_native if bond_variant == "native" else model.vision.norm_out(raw_native)
            else:
                pre_plus_pos = model.vision.features(imgs, return_pre_norm=True)  # (pre-norm h) + pos_emb
                pos_emb = model.vision.pos_emb
                post_reconstructed = model.vision.norm_out(pre_plus_pos - pos_emb) + pos_emb
                native_direct = model.vision.features(imgs)
                err = (post_reconstructed - native_direct).abs().max().item()
                assert err < 1e-4, f"conv norm-reconstruction mismatch: {err}"
                raw = post_reconstructed if bond_variant == "native" else pre_plus_pos
            return model.vis_proj(raw).detach()                                # (B, Nv, cfg.dim)

    def analyze(model, cfg, bond_variant):
        """bond_variant: 'native' (unchanged, real deployed bond) or 'alt' (the other
        convention: vit gets a final norm applied, conv has its final norm skipped), always at
        the same post-vis_proj bond odt_libero_action uses -- see get_leaf's docstring."""
        D = cfg.dim
        leaf_full = get_leaf(model, cfg, bond_variant)
        Nv = leaf_full.shape[1]; grid = int(round(Nv ** 0.5))

        def build_seq(vis, sl):
            B = vis.shape[0]
            bos = model.bos.expand(B, -1, -1)
            instr_e = model.tok_emb(instr[sl])
            st = model.state_proj(states[sl])[:, None]
            embe = model.embodiment_emb(emb0[sl])[:, None]
            aq = model.action_queries.expand(B, -1, -1)
            x = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
            return x + model.pos_emb[:, :x.shape[1]]

        def from_leaf(leaf_in, sl):
            x = build_seq(leaf_in, sl)
            mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
            h = model.backbone(x, mask=mask)
            h = model.norm_out(h)
            return model.action_head(h[:, -H:])

        G = torch.zeros(D, D, device=dev, dtype=torch.float64); n_rows = 0
        for i in range(0, leaf_full.shape[0], 256):
            sl = slice(i, i + 256)
            vb = leaf_full[sl].detach().requires_grad_(True)
            act = from_leaf(vb, sl)
            sel = act[:, :, dims].sum()
            gsel, = torch.autograd.grad(sel, vb)
            g = gsel.reshape(-1, D).double()
            G += g.T @ g; n_rows += g.shape[0]
        G = G / n_rows
        ev, V = torch.linalg.eigh(G)
        v_top = V.flip(1)[:, 0]

        rng_loc = torch.Generator(device=dev).manual_seed(0)
        k_top = max(1, int(round(0.1 * Nv)))

        @torch.no_grad()
        def patch_locality(v):
            proj = torch.einsum("bnd,d->bn", leaf_full.double(), v.double())
            energy = (proj ** 2).mean(0)
            return (torch.topk(energy, k_top).values.sum() / energy.sum().clamp_min(1e-30)).item()

        rand_locs = [patch_locality(torch.randn(D, generator=rng_loc, device=dev)) for _ in range(30)]
        top_loc = patch_locality(v_top)
        rand_mean = sum(rand_locs) / len(rand_locs)
        return {"top_causal_dir_locality": round(top_loc, 4), "random_dir_locality_mean": round(rand_mean, 4),
                "ratio": round(top_loc / max(rand_mean, 1e-9), 3), "Nv": Nv, "grid": grid}

    vit_model, vit_cfg = load_model(vit_ckpt, "vit")
    conv_model, conv_cfg = load_model(conv_ckpt, "conv")

    result = {"group": group, "n_eval": len(idx),
              "vit_native": analyze(vit_model, vit_cfg, "native"),
              "vit_alt_post_norm": analyze(vit_model, vit_cfg, "alt"),
              "conv_native_post_norm": analyze(conv_model, conv_cfg, "native"),
              "conv_alt_pre_norm": analyze(conv_model, conv_cfg, "alt"),
              "note": ("'native' = the real deployed bond for that encoder, unchanged; vit_alt "
                       "applies an extra final norm and conv_alt_pre_norm skips the conv norm. "
                       "Both alternate paths change the distribution seen by vis_proj and the "
                       "backbone. They are descriptive normalization-placement diagnostics, not "
                       "deployed-policy measurements or clean bond-isolation interventions."),
              }
    with open(f"{VOL_PATH}/odt_norm_bond_diagnostic_{group}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def decomposability_audit(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                          res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                          norm: str = "rational", n_batch: int = 512):
    """Does decomposability hold on the ACTUAL trained ~20M policy, not just architecturally?
    Three checks, all on real trained weights + real held-out data, no training/sim:
    (1) PURITY AUDIT: every nn.Module instance actually used in a forward pass belongs to the
        allowed {Linear, Embedding, BilinearFFN, BilinearAttention, RationalNorm, containers} —
        a mechanical certificate, not an architectural claim.
    (2) EXACT PROJECTIVE FOLD-AND-VERIFY on real weights: take one real backbone block's
        RationalNorm(pade) -> BilinearFFN branch, reconstruct it as a literal ratio of two
        polynomial tensor networks P(x)/Q(x) (BilinearFFN.dense_core() gives the exact cubic
        tensor T; the RationalNorm forward is algebraically rearranged into (numerator,
        denominator) form per DEVLOG cont.31's projective-coordinate construction), and verify
        the projective reconstruction reproduces the real forward pass to near machine precision
        in fp64 -- on the TRAINED weights and REAL activations, not a random toy net
        (rational_norm_proto.py) or an operator-level statistic (rational_norm_check).
    (3) FP64 vs FP32 whole-model determinism: the deployed model is a fixed algebraic function of
        its input, so raising precision should only shrink floating-point rounding, not reveal
        any hidden non-algebraic behavior (a branch, a lookup, a numerical solver)."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.bilinear import BilinearFFN
    from xvla.nn.normalization import RationalNorm, PerTokenRmsNorm, RmsBatchNorm, HomotopyNorm

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
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

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    print(f"loaded {ckpt}: dim={cfg.dim} d_a={d_a} n_samples={len(samples)}")

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_batch, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    out = {"ckpt": ckpt, "n_batch": len(idx)}

    # ---- (1) PURITY AUDIT ----
    allowed_leaf = {"Linear", "Embedding", "Conv2d", "BilinearFFN", "BilinearAttention",
                    "RationalNorm", "Parameter", "Dropout", "Identity"}
    forbidden_types = (PerTokenRmsNorm, RmsBatchNorm, HomotopyNorm)
    seen_types = {}
    forbidden_found = []
    for m in model.modules():
        tname = type(m).__name__
        is_leaf = len(list(m.children())) == 0
        if is_leaf:
            seen_types[tname] = seen_types.get(tname, 0) + 1
        if isinstance(m, forbidden_types):
            forbidden_found.append(tname)
        if is_leaf and tname not in allowed_leaf and "ModuleList" not in tname:
            forbidden_found.append(tname)
    # explicit negative check: no torch.nn.functional.softmax / GELU / LayerNorm modules
    disallowed_names = [n for n in seen_types if n in
                        ("Softmax", "GELU", "ReLU", "SiLU", "LayerNorm", "BatchNorm1d", "BatchNorm2d")]
    out["purity_audit"] = {
        "leaf_module_types_used": seen_types,
        "forbidden_norm_instances_found": forbidden_found,
        "disallowed_nonlinearity_modules_found": disallowed_names,
        "PASS": len(forbidden_found) == 0 and len(disallowed_names) == 0,
    }
    print("PURITY AUDIT:", json.dumps(out["purity_audit"], indent=2))

    # ---- (3) FP64 vs FP32 whole-model determinism (build the double copy first; also used by (2)) ----
    model64 = ChiVLA(cfg).double().to(dev)
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    model64.eval()
    with torch.no_grad():
        a32, _ = model(imgs, instr, states, emb0)
        a64, _ = model64(imgs.double(), instr, states.double(), emb0)
    d = (a64 - a32.double()).abs()
    out["fp64_vs_fp32"] = {
        "max_abs_diff": round(float(d.max()), 10),
        "mean_abs_diff": round(float(d.mean()), 10),
        "action_scale_for_reference": round(float(a64.abs().mean()), 6),
    }
    print("FP64 vs FP32:", json.dumps(out["fp64_vs_fp32"], indent=2))

    # ---- (2) EXACT PROJECTIVE FOLD-AND-VERIFY on real weights + real activations ----
    # "Direct" is computed NATIVELY in fp64 (model64), so the comparison isolates the algebraic
    # correctness of the projective reconstruction from ordinary fp32 rounding (matching the
    # rigor of the toy-net proto's ~1e-14 check, but on the real trained 20M-parameter policy).
    from xvla.nn.attention import causal_mask
    imgs64, instr64, states64, emb064 = imgs.double(), instr, states.double(), emb0
    with torch.no_grad():
        vis = model64._visual_tokens(imgs64)
        bos = model64.bos.expand(len(idx), -1, -1)
        instr_e = model64.tok_emb(instr64)
        st = model64.state_proj(states64)[:, None]
        embe = model64.embodiment_emb(emb064)[:, None]
        aq = model64.action_queries.expand(len(idx), -1, -1)
        x0 = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
        x0 = x0 + model64.pos_emb[:, :x0.shape[1]]
        mask = causal_mask(x0.shape[1], device=dev, dtype=x0.dtype)
        block0 = model64.backbone.blocks[0]
        u0 = block0.rbn_attn(x0)
        x1 = x0 + block0.attn_gain * block0.attn(u0, mask=mask)   # real residual stream, native fp64
    rbn: RationalNorm = block0.rbn_ffn
    ffn: BilinearFFN = block0.ffn
    assert rbn.variant == "pade" and not model64.training

    x64 = x1.reshape(-1, x1.shape[-1])                    # (B*N, dim) real trained-model activations, fp64
    with torch.no_grad():
        v_direct = rbn(x1).reshape(-1, x1.shape[-1])
        y_direct = ffn(rbn(x1)).reshape(-1, ffn.out_dim)

    eps_ = float(rbn.eps)
    s0 = rbn.running_ms.clamp_min(1e-12)
    pa = rbn.pa; pb = rbn.pb
    ms = x64.pow(2).mean(-1, keepdim=True) + eps_
    vv = ms / s0
    P_ = sum(pa[k] * vv ** k for k in range(rbn.deg + 1))
    Q_ = sum(pb[k] * vv ** k for k in range(rbn.deg + 1))
    N_ = x64 * P_ * torch.rsqrt(s0)                       # v_direct == N_ / Q_
    v_proj = N_ / Q_
    v_err = float((v_proj - v_direct).abs().max())

    h = torch.cat([Q_.expand(-1, 1), N_], dim=-1)          # h = Q_ * x̄(v), homogeneous, constant slot first
    T = ffn.dense_core()                                    # (out, dim+1, dim+1) exact cubic tensor, real weights, fp64
    # chunked to avoid materializing a (rows, out, dim+1, dim+1) intermediate (OOMs at full batch)
    y_num_parts = []
    for i in range(0, h.shape[0], 256):
        hc = h[i:i + 256]
        y_num_parts.append(torch.einsum("oij,bi,bj->bo", T, hc, hc))
    y_num = torch.cat(y_num_parts, dim=0)
    y_den = Q_ ** 2
    y_proj = y_num / y_den
    y_err_abs = float((y_proj - y_direct).abs().max())
    y_err_rel = float(((y_proj - y_direct).abs() / y_direct.abs().clamp_min(1e-10)).median())
    out["projective_fold_verify"] = {
        "site": "backbone.blocks[0].{rbn_ffn,ffn}", "batch_rows": int(x64.shape[0]),
        "precision_note": ("the projective (P/Q) reconstruction is computed in pure fp64 with no "
                           "internal downcast; the 'direct' reference calls the real deployed "
                           "RationalNorm.forward(), which by design internally computes its "
                           "per-instance statistic in fp32 (`xf = x.float()`) regardless of the "
                           "caller's dtype, so the residual below is bounded by that module's own "
                           "fp32 precision choice, not by any approximation in the rational algebra"),
        "norm_reconstruction_max_abs_err": v_err,
        "ffn_output_max_abs_err": y_err_abs,
        "ffn_output_median_rel_err": y_err_rel,
        "v_range_observed": [round(float(vv.min()), 4), round(float(vv.max()), 4)],
        "pade_fit_range": [0.1, 10.0],
    }
    print("PROJECTIVE FOLD-VERIFY:", json.dumps(out["projective_fold_verify"], indent=2))

    out["overall_decomposability_verified"] = (
        out["purity_audit"]["PASS"]
        and out["projective_fold_verify"]["ffn_output_median_rel_err"] < 1e-5   # ~ fp32 eps scale
    )
    with open(f"{VOL_PATH}/decomposability_audit_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(out, indent=2))
    return out


def compute_shaped_reward(obs, obs0, task_language, d0=0.15, d_thresh=0.06, h_max=0.03,
                          w_reach=1.0, w_place=1.5, w_lift=0.5, kappa=40.0):
    """NEXT_PHASE_PLAN.md section 2.3's potential-based reward shaping Phi(s) -- CPU-only, no
    torch/model call, standalone (the real production twin of xvla/train/awr_rl_proto.py's toy
    `phi()`). Reuses the EXACT object-filter + language-match already debugged in
    libero_causal_intervention/libero_causal_intervention_closedloop (obj_pos excludes
    robot0/basket/derived '_to_'/'eef' keys) -- that filter's one real bug (derived relational
    keys like 'X_to_robot0_eef_pos' masquerading as objects) is already fixed there and must not
    be re-derived here. Returns (phi, diag): phi is the scalar potential Phi(s); the CALLER (a
    rollout loop) computes F(s,a,s') = gamma*phi(s') - phi(s) - c_step for t < success, and must
    special-case r(T) = 1.0 with Phi treated as 0 at the absorbing success transition (Ng, Harada
    & Russell 1999's requirement) -- not handled here, since only the caller knows when that
    transition happens. `d0`/`d_thresh`/`h_max` should be calibrated from real successful
    trajectories (see `awr_calibrate_and_preflight`) rather than left at these generic defaults."""
    import re
    import numpy as np

    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    obj_pos = {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
              if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
              and "_to_" not in k and "eef" not in k}
    true_body = next((k for k in obj_pos if readable(k) in task_language.lower()), None)
    basket_key = next((k for k in obs if "basket" in k and k.endswith("_pos")), None)
    if true_body is None or basket_key is None:
        return 0.0, {"resolved": False, "true_body": true_body, "basket_key": basket_key}

    p_tgt = np.asarray(obj_pos[true_body]); p_basket = np.asarray(obs[basket_key])
    p_eef = np.asarray(obs["robot0_eef_pos"])
    q = np.asarray(obs["robot0_gripper_qpos"]); q0 = np.asarray(obs0["robot0_gripper_qpos"])
    p_tgt0 = np.asarray(obs0.get(f"{true_body}_pos", p_tgt))

    c = float(np.clip(1.0 - np.abs(q).mean() / (np.abs(q0).mean() + 1e-6), 0.0, 1.0))
    d_reach = float(np.linalg.norm(p_eef - p_tgt))
    d_place = float(np.linalg.norm(p_tgt - p_basket))
    h = float(p_tgt[2] - p_tgt0[2])
    g = c / (1.0 + np.exp(-kappa * (d_thresh - d_reach)))
    phi_reach = 1.0 - np.tanh(d_reach / d0)
    phi_place = 1.0 - np.tanh(d_place / d0)
    h_clip = float(np.clip(h, 0.0, h_max)) / h_max
    phi = float(w_reach * (1 - g) * phi_reach + w_place * g * phi_place + w_lift * h_clip)
    return phi, {"resolved": True, "true_body": true_body, "basket_key": basket_key,
                "c": round(c, 4), "d_reach": round(d_reach, 4), "d_place": round(d_place, 4),
                "h": round(h, 4), "g": round(float(g), 4)}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2400)
def awr_calibrate_and_preflight(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                                res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                                norm: str = "rational", max_task: int = 10, eps_per_task: int = 3,
                                n_steps: int = 280):
    """NEXT_PHASE_PLAN.md Part 2, staged plan steps 1+2: (1) pre-flight audit -- verify the
    language->object string-match `compute_shaped_reward` depends on resolves for every one of
    the 10 LIBERO-Object tasks BEFORE any training touches the reward; (2) calibrate the reward
    constants (d0, d_thresh, h_max) from real successful trajectories. HONEST DEVIATION from the
    plan text: the plan says calibrate from the OpenVLA teacher's trajectories, but that requires
    a ~14GB model download + slower per-step inference for a strictly WORSE success rate (88.9%
    teacher vs our own 94.5% checkpoint) purely to source calibration statistics that don't
    depend on which good policy produced them -- so this reuses our own already-validated,
    already-loaded checkpoint's successful closed-loop rollouts instead. Reports per-task
    resolution + the calibrated constants, ready to pass into compute_shaped_reward's defaults."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    d_a = frames[0][4].shape[0]; state_dim = frames[0][3].shape[0]
    A = np.stack([f[4] for f in frames]); S = np.stack([f[3] for f in frames])
    a_mu, a_sd = A.mean(0), A.std(0) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    print(f"loaded {ckpt}: d_a={d_a} state_dim={state_dim}")

    import re
    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    @torch.no_grad()
    def predict_chunk(obs, instr_str):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        instr_ids = torch.tensor([encode(instr_str)], device=dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()

    n_tasks = min(max_task, suite.n_tasks)
    preflight = {}
    d_reach0_all, d_place0_all, d_reach_at_close_all, h_lift_all = [], [], [], []
    n_success = n_total = 0

    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        for ep in range(eps_per_task):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            env.seed(ti * 100 + ep); obs0 = env.reset()
            for _ in range(10):
                obs0, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])

            obj_pos0 = {k.rsplit("_pos", 1)[0]: obs0[k] for k in obs0
                       if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
                       and "_to_" not in k and "eef" not in k}
            true_body = next((k for k in obj_pos0 if readable(k) in task.language.lower()), None)
            basket_key = next((k for k in obs0 if "basket" in k and k.endswith("_pos")), None)
            if ep == 0:
                preflight[ti] = {"language": task.language, "resolved": true_body is not None and basket_key is not None,
                                 "true_body": true_body, "basket_key": basket_key}
                print(f"[preflight task {ti}] {task.language} -> true_body={true_body} basket_key={basket_key}")
            if true_body is None or basket_key is None:
                env.close(); continue

            d_reach0 = float(np.linalg.norm(obs0["robot0_eef_pos"] - obj_pos0[true_body]))
            d_place0 = float(np.linalg.norm(obj_pos0[true_body] - obs0[basket_key]))
            d_reach0_all.append(d_reach0); d_place0_all.append(d_place0)

            q0 = np.asarray(obs0["robot0_gripper_qpos"])
            obs, ok, t, first_close_dreach, max_h = obs0, False, 0, None, 0.0
            z0 = float(obj_pos0[true_body][2])
            while t < n_steps and not ok:
                chunk = predict_chunk(obs, task.language)
                for kk in range(min(8, H, n_steps - t)):
                    a = chunk[kk].copy(); a[-1] = 1.0 if a[-1] > 0 else -1.0
                    obs, r, done, info = env.step(a.tolist()); t += 1
                    q = np.asarray(obs["robot0_gripper_qpos"])
                    c = float(np.clip(1.0 - np.abs(q).mean() / (np.abs(q0).mean() + 1e-6), 0.0, 1.0))
                    obj_pos_t = {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
                                if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
                                and "_to_" not in k and "eef" not in k}
                    if true_body in obj_pos_t:
                        d_r = float(np.linalg.norm(obs["robot0_eef_pos"] - obj_pos_t[true_body]))
                        if first_close_dreach is None and c > 0.5: first_close_dreach = d_r
                        max_h = max(max_h, float(obj_pos_t[true_body][2]) - z0)
                    if r > 0: ok = True
                    if done or ok or t >= n_steps: break
            env.close()
            n_total += 1; n_success += int(ok)
            if ok:
                if first_close_dreach is not None: d_reach_at_close_all.append(first_close_dreach)
                if max_h > 0: h_lift_all.append(max_h)
            print(f"[calib task {ti} ep {ep}] success={ok} d_reach0={d_reach0:.3f} d_place0={d_place0:.3f} "
                 f"first_close_dreach={first_close_dreach} max_h={max_h:.4f}")

    def pctl(xs, p): return round(float(np.percentile(xs, p)), 4) if xs else None
    calib = {
        "n_trials": n_total, "n_success": n_success,
        "success_rate": round(n_success / max(n_total, 1), 3),
        "d0_candidates": {"median_d_reach0": pctl(d_reach0_all, 50), "median_d_place0": pctl(d_place0_all, 50)},
        "d_thresh_candidate": pctl(d_reach_at_close_all, 50),
        "h_max_candidate": pctl(h_lift_all, 90),
        "recommended": {
            "d0": round(((pctl(d_reach0_all, 50) or 0.15) + (pctl(d_place0_all, 50) or 0.15)) / 2, 4),
            "d_thresh": pctl(d_reach_at_close_all, 50) or 0.06,
            "h_max": pctl(h_lift_all, 90) or 0.03,
        },
        "note": ("calibrated from our OWN 94.5% checkpoint's successful closed-loop rollouts "
                "(not the OpenVLA teacher, see docstring) -- d0 = median of (initial reach "
                "distance, initial target-to-basket distance)/2 (both should be normalized on a "
                "similar table-workspace scale); d_thresh = median eef-to-target distance at the "
                "first step gripper closedness c(t) crosses 0.5 on successful episodes; h_max = "
                "90th-percentile observed lift height on successful episodes (saturating fast, "
                "as the plan intends, not a ceiling estimate)."),
    }
    result = {"ckpt": ckpt, "preflight": preflight,
             "preflight_all_resolved": all(v["resolved"] for v in preflight.values()),
             "calibration": calib}
    with open(f"{VOL_PATH}/awr_calibrate_and_preflight_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=600)
def awr_product_smoke():
    """Cheap pre-flight smoke test for the new AWR-on-product-routing code (no LIBERO/HF
    download): synthetic random batch only, GPU, exercises every new code path
    (`ChiVLA(value_head=True)`, `pooled_features`, `ProductRoutingHead.sample_signs`/
    `.loss_awr`, the value-head foldability check, and one real backward+step of each new
    optimizer) so a bug is caught in ~a minute instead of after a multi-hour detached run."""
    _bootstrap()
    import torch
    import torch.nn.functional as F
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"
    B, H, d_a, state_dim, T = 8, 4, 7, 8, 16
    cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=64, vit_layers=2, vit_heads=4,
                    vocab_size=32, max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=64, n_layers=2, n_heads=4, action_horizon=H, action_dim=d_a,
                    action_head="product", n_factors=3, distill_teacher=True,
                    lambda_distill=1.0, curriculum=True, value_head=True)
    model = ChiVLA(cfg).to(dev)
    img = torch.rand(B, 3, 32, 32, device=dev)
    instr = torch.randint(0, 32, (B, T), device=dev)
    state = torch.randn(B, state_dim, device=dev)
    emb = torch.zeros(B, dtype=torch.long, device=dev)
    target = torch.randn(B, H, d_a, device=dev)

    # 1. forward + BC loss (curriculum + distill), backward through full model.
    actions, loss = model(img, instr, state, emb, target_actions=target, progress=0.3)
    assert actions.shape == (B, H, d_a), actions.shape
    loss.backward()
    print(f"[1] forward+BC loss OK: loss={loss.item():.4f} actions.shape={tuple(actions.shape)}")
    model.zero_grad(set_to_none=True)

    # 2. pooled_features == the same h forward() uses internally (spot check via decode()).
    with torch.no_grad():
        h = model.pooled_features(img, instr, state, emb)
        assert h.shape == (B, cfg.dim), h.shape
        dec = model.product_head.decode(h)
        assert dec.shape == (B, H, d_a), dec.shape
    print(f"[2] pooled_features/decode OK: h.shape={tuple(h.shape)}")

    # 3. sample_signs (no_grad exploration op) + loss_awr (real AWR gradient step).
    with torch.no_grad():
        a_sampled, b_sampled = model.product_head.sample_signs(h, temp=1.0)
        assert a_sampled.shape == (B, H * d_a) and b_sampled.shape == (B, cfg.n_factors)
        assert set(b_sampled.unique().tolist()) <= {-1.0, 1.0}
    fake_advantage = torch.randn(B, device=dev)
    weight = torch.exp(torch.clamp(fake_advantage, -5.0, 5.0) / 1.0)
    loss_awr = model.product_head.loss_awr(h, a_sampled, b_sampled, weight)
    opt_policy = torch.optim.AdamW(model.product_head.parameters(), lr=1e-3)
    opt_policy.zero_grad(set_to_none=True); loss_awr.backward(); opt_policy.step()
    print(f"[3] sample_signs+loss_awr OK: loss_awr={loss_awr.item():.4f} "
         f"b_sampled unique={sorted(set(b_sampled.unique().tolist()))}")

    # 4. value head: V(h), MSE regression step, distinct optimizer.
    v = model.value(h)
    assert v.shape == (B,), v.shape
    loss_v = F.mse_loss(v, fake_advantage)
    opt_value = torch.optim.AdamW(model.value_head.parameters(), lr=1e-3)
    opt_value.zero_grad(set_to_none=True); loss_v.backward(); opt_value.step()
    print(f"[4] value head OK: v.shape={tuple(v.shape)} loss_v={loss_v.item():.4f}")

    # 5. explicit foldability check: forward() output must be bit-identical with/without
    # value_head existing at all (checked, not assumed).
    model.eval()
    with torch.no_grad():
        out_before, _ = model(img, instr, state, emb)
    saved_vh = model.value_head
    del model.value_head
    with torch.no_grad():
        out_after, _ = model(img, instr, state, emb)
    model.value_head = saved_vh
    fold_ok = torch.equal(out_before, out_after)
    print(f"[5] foldability check (value_head absent vs present): {'PASS' if fold_ok else 'FAIL'}")
    assert fold_ok, "value_head must NEVER affect forward()'s deployed output"

    print("ALL SMOKE CHECKS PASSED")
    return {"ok": True, "loss_bc": loss.item(), "loss_awr": loss_awr.item(),
           "loss_value": loss_v.item(), "foldability": fold_ok}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=12 * 3600)
def awr_product_finetune(bc_steps: int = 2500, rounds: int = 2, eps_per_round: int = 8,
                         awr_iters_per_round: int = 250, eval_eps_per_task: int = 5,
                         n_frames: int = 20000, horizon: int = 8, res: int = 64,
                         max_task: int = 10, max_steps: int = 280, num_steps_wait: int = 10,
                         exec_h: int = 8, n_factors: int = 4, gamma: float = 0.99,
                         lam: float = 1.0, cap: float = 5.0, sample_temp: float = 1.0,
                         step_cost: float = 0.01, d0: float = 0.377, d_thresh: float = 0.043,
                         h_max_reward: float = 0.275, awr_batch: int = 64,
                         bc_batch: int = 256, value_lr: float = 3e-4, policy_lr: float = 3e-4,
                         bc_lr: float = 8e-4, norm: str = "per_token", seed: int = 0,
                         tag: str = "_awr_v1"):
    """AWR (advantage-weighted regression) fine-tuning of the product-routing multimodal head
    (NEXT_PHASE_PLAN.md Part 2, staged plan step 6/7; xvla/train/awr_product_proto.py is this
    function's NumPy dry-run, xvla/train/awr_rl_proto.py is the single-Gaussian baseline this
    is designed to beat). No product-routing checkpoint currently exists on the volume to
    resume from (checked: only ckpt_linear_rat_* files are present), so this BC-pretrains a
    product_fixed-recipe head from scratch (curriculum + distillation anchor, DEVLOG's winning
    ~72.5-82.5% recipe) THEN AWR-fine-tunes it -- one job, so the before/after AWR comparison
    below is measured in the exact same run with the exact same eval budget.

    Foldable-by-construction (checked, not assumed -- see the assertion after BC pretrain and
    again after AWR): `value_head` (xvla/models/vla.py, cfg.value_head) and
    `ProductRoutingHead.sample_signs` (the rollout-exploration op) are never read by
    `forward()`/`decode()`/`action_for_signs()` -- both are new, separate methods, and
    `forward()`'s own code is untouched (only pure-extracted into `_encode()`, zero behavior
    change). The AWR update (`ProductRoutingHead.loss_awr`) only touches `product_head`'s and
    `value_head`'s parameters; the vision+backbone stay FROZEN during the AWR phase (only the
    interleaved BC-anchor step below updates them) -- a deliberate sanity-run scoping choice
    (cheaper, avoids re-deriving h from raw pixels for every AWR minibatch), reported honestly,
    not hidden.

    Reward: `compute_shaped_reward`'s potential Phi(s), turned into a step reward via
    F(s,a,s') = gamma*Phi(s')-Phi(s)-step_cost for t<success, r(T)=1.0 at success (Ng/Harada/
    Russell 1999's boundary condition) -- computed over the executed exec_h-step macro-chunk and
    discounted within it, then a per-EPISODE Monte-Carlo return-to-go (gamma_macro = gamma**exec_h
    across macro-steps) is regressed by `value_head`; this is a simplification of the plan's
    TD(0)-with-truncation-bootstrap (honest: MC return-to-go, not a bootstrapped value target),
    acceptable for a first sanity pass at this budget.

    Training regime: BC-anchor step (weight=1, full model forward + product_head.loss with the
    same curriculum+distill config as pretraining) ALTERNATING with an AWR step (rollout-buffer
    minibatch, product_head.loss_awr + value MSE) each interleaved iteration -- RLPD-style
    50/50 mixing at the per-step granularity (not literally a mixed single batch, but equal-
    count alternation of demo and on-policy gradients)."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"; H = horizon
    torch.manual_seed(seed); np.random.seed(seed)
    name = "lerobot/libero_object_image"

    # ---- stage 0: load frames/vocab/normalization (identical to libero_rollout_head) ----
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
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"state_dim={state_dim} action_dim={d_a}")

    imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in samples], device=dev)
    states = torch.tensor((S - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="product", n_factors=n_factors, distill_teacher=True,
                    lambda_distill=1.0, curriculum=True, norm=norm, qk_norm=norm,
                    value_head=True)
    model = ChiVLA(cfg).to(dev)
    print(f"model params: {model.num_params():,}  (action_head=product n_factors={n_factors} "
          f"value_head=True curriculum=True distill_teacher=True lambda_distill=1.0)")

    # ---- stage 1: BC pretrain, product_fixed recipe (matches libero_rollout_head's loop) ----
    opt_full = torch.optim.AdamW(model.parameters(), lr=bc_lr, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=bc_lr, max_steps=bc_steps, warmup_frac=0.05)

    def bc_step(step, total_steps):
        for g in opt_full.param_groups: g["lr"] = _lr_at(step, tcfg)
        idx = torch.randint(len(samples), (bc_batch,), device=dev)
        opt_full.zero_grad(set_to_none=True)
        # BUG FIX (caught via `modal app logs` mid-run on the first launch, job stopped and
        # relaunched before it reached this phase): during the AWR-interleaved phase, `step`
        # keeps growing past `bc_steps`, so an unclamped `progress = step/total_steps` exceeds
        # 1.0 -- which flips `lambda_distill*(1-progress)` NEGATIVE inside ChiVLA.forward(),
        # actively pushing c0 AWAY from the teacher anchor instead of toward it. Clamp to 1.0
        # so every post-pretrain BC-anchor step keeps using the fully-converged (progress=1)
        # curriculum/distill config, not a sign-flipped one.
        progress = min(step / max(total_steps, 1), 1.0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs[idx], instr[idx], states[idx],
                            torch.zeros(bc_batch, dtype=torch.long, device=dev),
                            target_actions=actions[idx], progress=progress)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_full.step()
        return loss.item()

    model.train()
    for step in range(bc_steps + 1):
        l = bc_step(step, bc_steps)
        if step % 500 == 0: print(f"  [BC pretrain] step {step} loss {l:.4f}")
    model.eval()
    torch.save(model.state_dict(), f"{VOL_PATH}/ckpt_product_bc{tag}.pt")
    print(f"saved pre-AWR BC checkpoint -> ckpt_product_bc{tag}.pt")

    # ---- explicit foldability check (checked, not assumed): forward()'s deployed output must
    # be bit-identical whether or not value_head exists. Mirrors awr_rl_proto.py's
    # check_foldability, done on the REAL model+real weights, not a toy analog. ----
    def check_value_head_foldability():
        idx = torch.randint(len(samples), (8,), device=dev)
        eids = torch.zeros(8, dtype=torch.long, device=dev)
        with torch.no_grad():
            out_before, _ = model(imgs[idx], instr[idx], states[idx], eids)
        saved_vh = model.value_head
        del model.value_head          # simulate "value head doesn't exist"
        try:
            with torch.no_grad():
                out_after, _ = model(imgs[idx], instr[idx], states[idx], eids)
            ok = torch.equal(out_before, out_after)
        finally:
            model.value_head = saved_vh
        print(f"FOLDABILITY CHECK (value_head absent vs present, forward() output): "
             f"{'PASS' if ok else 'FAIL'}")
        return ok

    fold_ok_pre_awr = check_value_head_foldability()

    # ---- quick pre-AWR closed-loop eval (same short budget used post-AWR below, for a fair
    # same-run before/after comparison; NOT a final capability number) ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = min(max_task, suite.n_tasks)
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    @torch.no_grad()
    def eval_closedloop(eps_per_task, label):
        per_task = {}
        for ti in range(n_tasks):
            task = suite.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            instr_ids = torch.tensor([encode(task.language)], device=dev)
            init_states = _libero_init_states_or_raise(suite, ti)
            succ = 0
            for ep in range(eps_per_task):
                env.seed(ti * 100 + ep); obs = env.reset()
                if init_states is not None:
                    obs = env.set_init_state(init_states[ep % len(init_states)])
                    for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
                ok = False; t = 0
                while t < max_steps and not ok:
                    im, st = obs_tensors(obs)
                    a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                    chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
                    for k in range(min(exec_h, H)):
                        act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                        obs, r, done, info = env.step(act.tolist()); t += 1
                        if r > 0: ok = True
                        if done or ok or t >= max_steps: break
                succ += int(ok)
            env.close()
            per_task[task.language] = round(succ / eps_per_task, 3)
            print(f"[{label} task {ti}] {task.language}: {succ}/{eps_per_task}")
        return {"per_task": per_task, "overall": round(float(np.mean(list(per_task.values()))), 3)}

    model.eval()
    pre_awr_eval = eval_closedloop(eval_eps_per_task, "pre-AWR (BC only)")
    print("PRE-AWR EVAL:", json.dumps(pre_awr_eval, indent=2))

    # ---- stage 2: rollout collection + interleaved AWR/BC fine-tune ----
    # Two separate optimizers (distinct LRs, no shared graph between the two backward calls
    # per iteration below) rather than one AdamW with param groups -- simpler to get right.
    opt_value = torch.optim.AdamW(model.value_head.parameters(), lr=value_lr,
                                  betas=(0.9, 0.95), weight_decay=0.0)
    opt_policy = torch.optim.AdamW(model.product_head.parameters(), lr=policy_lr,
                                   betas=(0.9, 0.95), weight_decay=0.0)

    gamma_macro = gamma ** exec_h
    round_logs = []
    for rnd in range(rounds):
        # ---- collect eps_per_round on-policy episodes with the CURRENT product_head, using
        # the exploration op sample_signs (temp=sample_temp) -- the executed vertex, not a
        # deterministic decode, so there is real routing exploration to learn from. ----
        model.eval()
        buf_h, buf_b, buf_a, buf_R = [], [], [], []
        n_success = 0
        for ep in range(eps_per_round):
            ti = (rnd * eps_per_round + ep) % n_tasks
            task = suite.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            instr_ids = torch.tensor([encode(task.language)], device=dev)
            init_states = _libero_init_states_or_raise(suite, ti)
            env.seed(ti * 1000 + rnd * 37 + ep); obs = env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
            obs0 = obs
            ep_h, ep_b, ep_a, ep_r = [], [], [], []
            ok = False; t = 0
            while t < max_steps and not ok:
                im, st = obs_tensors(obs)
                with torch.no_grad():
                    h = model.pooled_features(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                    a_sampled, b_sampled = model.product_head.sample_signs(h, temp=sample_temp)
                chunk = (a_sampled[0].reshape(H, d_a) * a_sd_t + a_mu_t).cpu().numpy()
                r_acc = 0.0
                for k in range(min(exec_h, H)):
                    phi_cur, _ = compute_shaped_reward(obs, obs0, task.language, d0, d_thresh, h_max_reward)
                    act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                    obs_next, r_env, done, info = env.step(act.tolist()); t += 1
                    if r_env > 0:
                        # REWARD FIX (cont.68 diagnostic): the terminal transition must be
                        # r_env + gamma*0 - phi_cur (Phi(absorbing)=0), not a bare 1.0 -- the old
                        # code implicitly assumed phi_cur was already ~0 right before success,
                        # but phi is generally near its MAXIMUM there, so every successful episode
                        # got an uncontrolled, uncancelled bonus of roughly +phi_cur relative to
                        # the theory-compliant construction. Subtracting phi_cur here restores the
                        # Ng/Harada/Russell boundary condition the docstring always claimed to use.
                        r_step = r_env - phi_cur; ok = True
                    else:
                        phi_next, _ = compute_shaped_reward(obs_next, obs0, task.language, d0, d_thresh, h_max_reward)
                        r_step = gamma * phi_next - phi_cur - step_cost
                    r_acc += (gamma ** k) * r_step
                    obs = obs_next
                    if done or ok or t >= max_steps: break
                ep_h.append(h[0].detach()); ep_b.append(b_sampled[0].detach())
                ep_a.append(a_sampled[0].detach()); ep_r.append(r_acc)
                if done or ok or t >= max_steps: break
            env.close()
            n_success += int(ok)
            # Monte-Carlo return-to-go across this episode's macro-steps.
            R = 0.0; returns = [0.0] * len(ep_r)
            for i in range(len(ep_r) - 1, -1, -1):
                R = ep_r[i] + gamma_macro * R
                returns[i] = R
            buf_h.extend(ep_h); buf_b.extend(ep_b); buf_a.extend(ep_a); buf_R.extend(returns)
        print(f"[round {rnd}] collected {len(buf_h)} macro-steps from {eps_per_round} episodes "
             f"({n_success}/{eps_per_round} sparse-success), return-to-go range "
             f"[{min(buf_R):.3f}, {max(buf_R):.3f}]")

        H_buf = torch.stack(buf_h)                     # (N, dim)
        B_buf = torch.stack(buf_b)                      # (N, G)
        A_buf = torch.stack(buf_a)                       # (N, m)
        R_buf = torch.tensor(buf_R, dtype=torch.float32, device=dev)

        model.train()
        for it in range(awr_iters_per_round):
            # -- BC-anchor step (weight=1, full model, curriculum+distill; updates all params) --
            l_bc = bc_step(bc_steps + rnd * awr_iters_per_round + it, bc_steps)

            # -- AWR step (rollout buffer minibatch; product_head + value_head only) --
            bidx = torch.randint(len(buf_h), (min(awr_batch, len(buf_h)),), device=dev)
            h_b, b_b, a_b, R_b = H_buf[bidx], B_buf[bidx], A_buf[bidx], R_buf[bidx]
            v = model.value(h_b)
            advantage = (R_b - v.detach())
            weight = torch.exp(torch.clamp(advantage, -cap, cap) / lam)
            loss_policy = model.product_head.loss_awr(h_b, a_b, b_b, weight)
            loss_value = F.mse_loss(v, R_b)
            opt_policy.zero_grad(set_to_none=True); opt_value.zero_grad(set_to_none=True)
            (loss_policy + loss_value).backward()
            opt_policy.step(); opt_value.step()

            if it % 50 == 0:
                print(f"  [round {rnd} it {it}] bc_loss={l_bc:.4f} awr_policy_loss={loss_policy.item():.4f} "
                     f"value_loss={loss_value.item():.4f} mean_adv={advantage.mean().item():.4f} "
                     f"mean_weight={weight.mean().item():.4f}")
        round_logs.append({"round": rnd, "n_success_rollout": n_success, "n_episodes": eps_per_round,
                           "buffer_size": len(buf_h), "final_bc_loss": l_bc,
                           "final_policy_loss": loss_policy.item(), "final_value_loss": loss_value.item()})

    model.eval()
    fold_ok_post_awr = check_value_head_foldability()
    torch.save(model.state_dict(), f"{VOL_PATH}/ckpt_product_awr{tag}.pt")
    print(f"saved post-AWR checkpoint -> ckpt_product_awr{tag}.pt")

    post_awr_eval = eval_closedloop(eval_eps_per_task, "post-AWR")
    print("POST-AWR EVAL:", json.dumps(post_awr_eval, indent=2))

    result = {
        "config": {"bc_steps": bc_steps, "rounds": rounds, "eps_per_round": eps_per_round,
                  "awr_iters_per_round": awr_iters_per_round, "n_factors": n_factors,
                  "gamma": gamma, "lam": lam, "cap": cap, "sample_temp": sample_temp,
                  "d0": d0, "d_thresh": d_thresh, "h_max_reward": h_max_reward,
                  "norm": norm, "seed": seed, "tag": tag},
        "foldability": {"pre_awr": fold_ok_pre_awr, "post_awr": fold_ok_post_awr},
        "pre_awr_eval": pre_awr_eval, "post_awr_eval": post_awr_eval,
        "round_logs": round_logs,
        "eval_eps_per_task": eval_eps_per_task,
        "note": ("SANITY RUN, not a final number: eval_eps_per_task is small (variance is "
                "large at this budget, matching this project's own documented seed-variance "
                "finding for the product head). Backbone/vision were FROZEN during the AWR "
                "phase (only BC-anchor steps update them); only product_head+value_head were "
                "AWR-updated. Advantage uses Monte-Carlo return-to-go, not a bootstrapped "
                "TD(0) value target."),
    }
    with open(f"{VOL_PATH}/awr_product_finetune{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({"pre_awr_eval": pre_awr_eval, "post_awr_eval": post_awr_eval,
                                 "foldability": result["foldability"]}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3600)
def awr_reward_diagnostic(bc_ckpt: str = "ckpt_product_bc_awr_pilot.pt",
                          post_ckpt: str = "ckpt_product_awr_awr_pilot.pt",
                          eps_per_task: int = 3, n_frames: int = 20000, res: int = 64,
                          horizon: int = 8, max_task: int = 10, max_steps: int = 280,
                          num_steps_wait: int = 10, exec_h: int = 8, n_factors: int = 4,
                          gamma: float = 0.99, sample_temp: float = 1.0, step_cost: float = 0.01,
                          d0: float = 0.377, d_thresh: float = 0.043, h_max_reward: float = 0.275,
                          norm: str = "per_token", seed: int = 0, grasp_c_thresh: float = 0.5,
                          shove_g_thresh: float = 0.8, shove_h_thresh: float = 0.1):
    """Cheap, NO-RETRAINING instrumented diagnostic requested directly in response to the reward-
    design-audit workflow's synthesis (3 ranked hypotheses for why AWR made the product head
    WORSE 3/3 times). `compute_shaped_reward` already computes and returns a full diagnostic dict
    (g, c, d_reach, d_place, h) but the AWR training loop (awr_product_finetune) never logs it --
    this function reuses the EXACT same model config / frame-loading / reward code and reruns
    rollouts (sample_signs exploration, matching what AWR itself trained on, not greedy decode)
    against two already-saved checkpoints from the real awr_pilot run (89.3%->86.7%) -- the
    pre-AWR BC checkpoint and the post-AWR checkpoint -- purely to LOG what the audit could only
    reason about from code. Tests, in one pass, all three ranked hypotheses:

    (1) GATE DISCONTINUITY at grasp: for every substep where `c` (gripper-closedness proxy)
        crosses grasp_c_thresh=0.5 upward, record phi_reach/phi_place/r_step immediately before
        and after -- the audit predicted a ~-0.5 reward spike exactly at this transition.
    (2) SHOVE SIGNATURE (grasp-blind phi_place): for every substep where the gate g exceeds
        shove_g_thresh=0.8 (i.e. phi is dominated by w_place), record whether h_clip stays below
        shove_h_thresh=0.1 (object never actually lifted) -- a high fraction here is the audit's
        predicted "reward without a real grasp" exploit signature.
    (3) STEP_COST DOMINANCE: per episode, record length T, accumulated step_cost*T, final
        Monte-Carlo return-to-go R, and whether r_env ultimately succeeded -- checks whether R's
        SIGN correlates more with episode length than with actual success.

    Also logs `resolved` (the language->object string-match flag) per substep -- the
    grounding-coverage lens's hypothesis (silent phi=0.0 on failed resolution) degenerated to a
    placeholder in the audit workflow and was never actually checked; this closes that gap too.

    Runs `eps_per_task` episodes per task (both checkpoints, so 2x eps_per_task*max_task total),
    reusing the identical model cfg / frame cache / normalization stats awr_product_finetune uses
    (so d0/d_thresh/h_max_reward calibration is evaluated in the exact setting it was tuned for),
    with NO gradient updates anywhere -- pure forward-pass rollout + reward instrumentation."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"; H = horizon
    torch.manual_seed(seed); np.random.seed(seed)
    name = "lerobot/libero_object_image"

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

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="product", n_factors=n_factors, distill_teacher=True,
                    lambda_distill=1.0, curriculum=True, norm=norm, qk_norm=norm,
                    value_head=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = min(max_task, suite.n_tasks)
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    @torch.no_grad()
    def instrumented_rollout(model, label):
        gamma_macro = gamma ** exec_h
        episodes = []
        grasp_transitions = []
        shove_substeps = 0; high_g_substeps = 0; unresolved_substeps = 0; total_substeps = 0
        for ti in range(n_tasks):
            task = suite.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            instr_ids = torch.tensor([encode(task.language)], device=dev)
            init_states = _libero_init_states_or_raise(suite, ti)
            for ep in range(eps_per_task):
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
                env.seed(ti * 1000 + ep); obs = env.reset()
                if init_states is not None:
                    obs = env.set_init_state(init_states[ep % len(init_states)])
                    for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
                obs0 = obs
                prev_c_below = True   # gripper starts open
                ep_r = []; t = 0; ok = False
                ep_step_cost_total = 0.0
                while t < max_steps and not ok:
                    im, st = obs_tensors(obs)
                    h_feat = model.pooled_features(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                    a_sampled, _ = model.product_head.sample_signs(h_feat, temp=sample_temp)
                    chunk = (a_sampled[0].reshape(H, d_a) * a_sd_t + a_mu_t).cpu().numpy()
                    r_acc = 0.0
                    for k in range(min(exec_h, H)):
                        phi_cur, diag_cur = compute_shaped_reward(obs, obs0, task.language, d0, d_thresh, h_max_reward)
                        act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                        obs_next, r_env, done, info = env.step(act.tolist()); t += 1
                        total_substeps += 1
                        if not diag_cur.get("resolved", True): unresolved_substeps += 1
                        g_cur = diag_cur.get("g", 0.0); c_cur = diag_cur.get("c", 0.0)
                        if g_cur > shove_g_thresh:
                            high_g_substeps += 1
                            h_lift = diag_cur.get("h", 0.0)
                            h_clip_cur = min(max(h_lift, 0.0), h_max_reward) / h_max_reward
                            if h_clip_cur < shove_h_thresh: shove_substeps += 1
                        crossed_up = prev_c_below and c_cur >= grasp_c_thresh
                        prev_c_below = c_cur < grasp_c_thresh
                        if r_env > 0:
                            r_step = 1.0; ok = True
                            phi_next, diag_next = phi_cur, diag_cur
                        else:
                            phi_next, diag_next = compute_shaped_reward(obs_next, obs0, task.language, d0, d_thresh, h_max_reward)
                            r_step = gamma * phi_next - phi_cur - step_cost
                            ep_step_cost_total += step_cost
                        if crossed_up:
                            grasp_transitions.append({
                                "task": ti, "ep": ep, "t": t,
                                "phi_reach_before": None, "phi_place_before": None,
                                "d_reach": diag_cur.get("d_reach"), "d_place": diag_cur.get("d_place"),
                                "phi_before": round(phi_cur, 4), "phi_after": round(phi_next, 4),
                                "r_step_at_transition": round(r_step, 4), "g_before": round(g_cur, 4),
                            })
                        r_acc += (gamma ** k) * r_step
                        obs = obs_next
                        if done or ok or t >= max_steps: break
                    ep_r.append(r_acc)
                    if done or ok or t >= max_steps: break
                env.close()
                R = 0.0
                for i in range(len(ep_r) - 1, -1, -1): R = ep_r[i] + gamma_macro * R
                episodes.append({"task": ti, "ep": ep, "length": t, "success": bool(ok),
                                 "return_to_go_R": round(R, 4),
                                 "accumulated_step_cost": round(ep_step_cost_total, 4)})
            print(f"[{label} task {ti}] {task.language}: done")

        n_ep = len(episodes)
        succ_eps = [e for e in episodes if e["success"]]
        fail_eps = [e for e in episodes if not e["success"]]
        summary = {
            "label": label, "n_episodes": n_ep,
            "n_success": len(succ_eps),
            "mean_length_success": round(float(np.mean([e["length"] for e in succ_eps])), 2) if succ_eps else None,
            "mean_length_fail": round(float(np.mean([e["length"] for e in fail_eps])), 2) if fail_eps else None,
            "mean_R_success": round(float(np.mean([e["return_to_go_R"] for e in succ_eps])), 4) if succ_eps else None,
            "mean_R_fail": round(float(np.mean([e["return_to_go_R"] for e in fail_eps])), 4) if fail_eps else None,
            "frac_success_with_negative_R": round(float(np.mean([e["return_to_go_R"] < 0 for e in succ_eps])), 4) if succ_eps else None,
            "corr_length_vs_R": round(float(np.corrcoef([e["length"] for e in episodes],
                                                         [e["return_to_go_R"] for e in episodes])[0, 1]), 4) if n_ep > 2 else None,
            "n_grasp_transitions": len(grasp_transitions),
            "mean_r_step_at_grasp_transition": round(float(np.mean([g["r_step_at_transition"] for g in grasp_transitions])), 4) if grasp_transitions else None,
            "frac_grasp_transitions_with_negative_r_step": round(float(np.mean([g["r_step_at_transition"] < 0 for g in grasp_transitions])), 4) if grasp_transitions else None,
            "total_substeps": total_substeps,
            "high_g_substeps": high_g_substeps,
            "shove_substeps_among_high_g": shove_substeps,
            "shove_frac_of_high_g": round(shove_substeps / max(high_g_substeps, 1), 4),
            "unresolved_substeps": unresolved_substeps,
            "unresolved_frac": round(unresolved_substeps / max(total_substeps, 1), 4),
        }
        return {"summary": summary, "episodes": episodes, "grasp_transitions": grasp_transitions[:200]}

    results = {}
    for label, ckpt_name in [("pre_awr", bc_ckpt), ("post_awr", post_ckpt)]:
        model = ChiVLA(cfg).to(dev)
        model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_name}", map_location=dev, weights_only=True))
        model.eval()
        print(f"=== {label} ({ckpt_name}) ===")
        results[label] = instrumented_rollout(model, label)
        print(f"[{label}] summary:", json.dumps(results[label]["summary"], indent=2))
        del model; torch.cuda.empty_cache()

    out = {"config": {"bc_ckpt": bc_ckpt, "post_ckpt": post_ckpt, "eps_per_task": eps_per_task,
                      "d0": d0, "d_thresh": d_thresh, "h_max_reward": h_max_reward,
                      "step_cost": step_cost, "gamma": gamma, "sample_temp": sample_temp,
                      "grasp_c_thresh": grasp_c_thresh, "shove_g_thresh": shove_g_thresh,
                      "shove_h_thresh": shove_h_thresh},
          "pre_awr": results["pre_awr"]["summary"], "post_awr": results["post_awr"]["summary"],
          "pre_awr_grasp_transitions": results["pre_awr"]["grasp_transitions"],
          "post_awr_grasp_transitions": results["post_awr"]["grasp_transitions"],
          "honest_note": ("Diagnostic-only, zero gradient updates, reusing sample_signs exploration "
                          "(matching what AWR itself trained on) against the two already-saved "
                          "checkpoints from the real awr_pilot run. Tests the reward-design-audit "
                          "workflow's 3 ranked hypotheses (gate discontinuity at grasp, grasp-blind "
                          "shove signature, step_cost length-dominance) plus the grounding-coverage "
                          "lens's unresolved-substep hypothesis that the audit workflow itself failed "
                          "to check (degenerate placeholder output).")}
    with open(f"{VOL_PATH}/awr_reward_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print("DIAGNOSTIC RESULT:", json.dumps({"pre_awr": out["pre_awr"], "post_awr": out["post_awr"]}, indent=2))
    return {"pre_awr": out["pre_awr"], "post_awr": out["post_awr"]}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def libero_causal_intervention(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                               res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                               norm: str = "rational", max_task: int = 8, eps_per_task: int = 4):
    """A REAL causal intervention on the actual trained, closed-loop-competitive policy, in the
    live simulator (not an offline proxy metric). Every LIBERO-Object scene contains several
    distractor grocery items alongside the true target (libero_scene_objects confirms this).
    For each (task, episode), with image and robot state held FIXED: (a) run the model with the
    TRUE instruction and the CONTROL instruction naming a co-present DISTRACTOR object; (b)
    compare each predicted first-step reach direction against the ground-truth direction from
    the gripper to the true target and to the distractor (both known exactly from the sim state).
    If the mechanism is causal, the predicted direction should track WHICHEVER object is named,
    not always the demo's original target -- the same test as the paper's synthetic-task
    counterfactual (R^2=0.956, switch-rate 97.1%), now on the real robot policy."""
    _bootstrap()
    import json, os, re
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    import pickle
    frames = pickle.load(open(cache, "rb"))
    # a_mu/a_sd/s_mu/s_sd MUST exactly match what the checkpoint was trained with (a per-dim mean
    # /std pooled over every individual raw per-step action/state, identical to how
    # libero_rollout_head computes them via A.mean((0,1)) over (n_samples,H,d_a) chunks -- pooling
    # over the horizon axis there is equivalent to pooling over all individual per-step actions
    # here, since it's the same underlying set of raw actions/states, just grouped differently).
    d_a = frames[0][4].shape[0]; state_dim = frames[0][3].shape[0]
    A = np.stack([f[4] for f in frames]); S = np.stack([f[3] for f in frames])
    a_mu, a_sd = A.mean(0), A.std(0) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    print(f"loaded {ckpt}: d_a={d_a} state_dim={state_dim}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def readable(obj_body):    # 'salad_dressing_1' -> 'salad dressing'
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    @torch.no_grad()
    def predict_first_delta(obs, instr_str):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        instr_ids = torch.tensor([encode(instr_str)], device=dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        raw = (a[0] * a_sd_t + a_mu_t).cpu().numpy()       # (H, d_a) real physical delta command
        return raw[0, :3]                                    # first-step translation delta

    n_tasks = min(max_task, suite.n_tasks)
    rows = []
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        for ep in range(eps_per_task):
            env.seed(ti * 100 + ep)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])   # settle
            # robosuite obs also carries DERIVED relational keys like 'X_to_robot0_eef_pos'
            # (object pose relative to the gripper) alongside the true 'X_pos' absolute position
            # -- exclude those, or "distractors" end up being bogus non-object relational entries
            # whose "readable" name is out-of-vocabulary noise, not a real counterfactual target.
            obj_pos = {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
                      if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
                      and "_to_" not in k and "eef" not in k}
            true_body = next((k for k in obj_pos if readable(k) in task.language.lower()), None)
            distractors = [k for k in obj_pos if k != true_body]
            if true_body is None or not distractors:
                env.close(); continue
            cf_body = distractors[ep % len(distractors)]
            eef = obs["robot0_eef_pos"]
            gt_true = obj_pos[true_body] - eef; gt_true = gt_true / (np.linalg.norm(gt_true) + 1e-8)
            gt_cf = obj_pos[cf_body] - eef; gt_cf = gt_cf / (np.linalg.norm(gt_cf) + 1e-8)
            instr_true = task.language
            instr_cf = f"pick up the {readable(cf_body)} and place it in the basket"
            d_true_instr = predict_first_delta(obs, instr_true)
            d_true_instr = d_true_instr / (np.linalg.norm(d_true_instr) + 1e-8)
            d_cf_instr = predict_first_delta(obs, instr_cf)
            d_cf_instr = d_cf_instr / (np.linalg.norm(d_cf_instr) + 1e-8)
            cos = lambda a_, b_: float(np.dot(a_, b_))
            row = {"task": ti, "ep": ep, "true_obj": true_body, "cf_obj": cf_body,
                   "under_true_instr": {"cos_to_true": round(cos(d_true_instr, gt_true), 3),
                                        "cos_to_cf": round(cos(d_true_instr, gt_cf), 3)},
                   "under_cf_instr": {"cos_to_true": round(cos(d_cf_instr, gt_true), 3),
                                     "cos_to_cf": round(cos(d_cf_instr, gt_cf), 3)}}
            row["switched"] = (row["under_true_instr"]["cos_to_true"] > row["under_true_instr"]["cos_to_cf"]
                               and row["under_cf_instr"]["cos_to_cf"] > row["under_cf_instr"]["cos_to_true"])
            rows.append(row)
            print(f"[task {ti} ep {ep}] true={true_body} cf={cf_body} switched={row['switched']}", row)
        env.close()

    switch_rate = round(float(np.mean([r["switched"] for r in rows])), 3) if rows else None
    mean_cos_true_under_true = round(float(np.mean([r["under_true_instr"]["cos_to_true"] for r in rows])), 3)
    mean_cos_cf_under_cf = round(float(np.mean([r["under_cf_instr"]["cos_to_cf"] for r in rows])), 3)
    mean_cos_true_under_cf = round(float(np.mean([r["under_cf_instr"]["cos_to_true"] for r in rows])), 3)
    result = {"ckpt": ckpt, "n_trials": len(rows), "switch_rate": switch_rate,
              "mean_cos_to_named_object": {"true_instr": mean_cos_true_under_true,
                                           "cf_instr": mean_cos_cf_under_cf},
              "mean_cos_to_original_target_under_cf_instr": mean_cos_true_under_cf,
              "rows": rows}
    with open(f"{VOL_PATH}/libero_causal_intervention_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2400)
def libero_causal_intervention_closedloop(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                                          res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                                          norm: str = "rational", max_task: int = 8, eps_per_task: int = 4,
                                          n_steps: int = 40, exec_h: int = 8):
    """NEXT_PHASE_PLAN.md Part 1, item 1.1: the multi-step / closed-loop fix to the honest-negative
    single-step instruction-swap test (libero_causal_intervention, switch-rate 15.6%). Same (task,
    distractor) setup, but now runs a REAL receding-horizon rollout (predict_chunk, exec_h=8) for
    n_steps with the instruction held FIXED throughout, under both the true and the counterfactual
    instruction, and tracks eef-to-object distance over time -- does the weak step-0 signal
    strengthen, decay, or stay flat over a real episode? A closed-loop, Euclidean-proximity delivery
    proxy (NOT LIBERO's own bddl containment predicate, which is target-specific and doesn't
    recognize a counterfactual object -- this is an honest simplification, reported as such): does
    the gripper end up closer to the NAMED object than to the original target by the end."""
    _bootstrap()
    import json, os, re, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    d_a = frames[0][4].shape[0]; state_dim = frames[0][3].shape[0]
    A = np.stack([f[4] for f in frames]); S = np.stack([f[3] for f in frames])
    a_mu, a_sd = A.mean(0), A.std(0) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    print(f"loaded {ckpt}: d_a={d_a} state_dim={state_dim}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    @torch.no_grad()
    def predict_chunk(obs, instr_str):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        instr_ids = torch.tensor([encode(instr_str)], device=dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()

    def rollout_fixed_instr(env, instr_str, seed):
        env.seed(seed); obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
        dists_true, dists_cf, t = [], [], 0
        while t < n_steps:
            chunk = predict_chunk(obs, instr_str)
            for k in range(min(exec_h, H, n_steps - t)):
                a = chunk[k].copy(); a[-1] = 1.0 if a[-1] > 0 else -1.0
                obs, r, done, info = env.step(a.tolist()); t += 1
                dists_true.append(float(np.linalg.norm(obs["robot0_eef_pos"] - obj_pos_cache["true"])))
                dists_cf.append(float(np.linalg.norm(obs["robot0_eef_pos"] - obj_pos_cache["cf"])))
                if done or t >= n_steps: break
        return dists_true, dists_cf

    n_tasks = min(max_task, suite.n_tasks)
    rows = []
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        for ep in range(eps_per_task):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            seed = ti * 100 + ep
            env.seed(seed); obs0 = env.reset()
            obj_pos = {k.rsplit("_pos", 1)[0]: obs0[k] for k in obs0
                      if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
                      and "_to_" not in k and "eef" not in k}
            true_body = next((k for k in obj_pos if readable(k) in task.language.lower()), None)
            distractors = [k for k in obj_pos if k != true_body]
            env.close()
            if true_body is None or not distractors:
                continue
            cf_body = distractors[ep % len(distractors)]
            obj_pos_cache = {"true": obj_pos[true_body], "cf": obj_pos[cf_body]}
            instr_cf = f"pick up the {readable(cf_body)} and place it in the basket"

            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            dt_true_run, dc_true_run = rollout_fixed_instr(env, task.language, seed)
            env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            dt_cf_run, dc_cf_run = rollout_fixed_instr(env, instr_cf, seed)
            env.close()

            row = {"task": ti, "ep": ep, "true_obj": true_body, "cf_obj": cf_body,
                   "true_instr_run": {"dist_to_true_start": round(dt_true_run[0], 4), "dist_to_true_end": round(dt_true_run[-1], 4),
                                     "dist_to_cf_start": round(dc_true_run[0], 4), "dist_to_cf_end": round(dc_true_run[-1], 4)},
                   "cf_instr_run": {"dist_to_true_start": round(dt_cf_run[0], 4), "dist_to_true_end": round(dt_cf_run[-1], 4),
                                   "dist_to_cf_start": round(dc_cf_run[0], 4), "dist_to_cf_end": round(dc_cf_run[-1], 4)}}
            # closed-loop "switched": under the CF instruction, did the gripper end up CLOSER to the
            # newly-named object than to the original target, having started closer to the original
            # (a real trajectory-level delivery-direction proxy, not a single-step direction).
            row["cf_run_ended_closer_to_named"] = dc_cf_run[-1] < dt_cf_run[-1]
            row["cf_run_improved_toward_named"] = (dc_cf_run[-1] - dc_cf_run[0]) < (dt_cf_run[-1] - dt_cf_run[0])
            rows.append(row)
            print(f"[task {ti} ep {ep}] true={true_body} cf={cf_body} "
                  f"cf_run end dist(true={dt_cf_run[-1]:.3f}, cf={dc_cf_run[-1]:.3f})", row)

    n = len(rows)
    result = {
        "ckpt": ckpt, "n_trials": n, "n_steps": n_steps,
        "frac_cf_run_ended_closer_to_named_object": round(float(np.mean([r["cf_run_ended_closer_to_named"] for r in rows])), 3) if n else None,
        "frac_cf_run_improved_toward_named_object": round(float(np.mean([r["cf_run_improved_toward_named"] for r in rows])), 3) if n else None,
        "mean_dist_to_true_under_true_instr": {"start": round(float(np.mean([r["true_instr_run"]["dist_to_true_start"] for r in rows])), 4),
                                               "end": round(float(np.mean([r["true_instr_run"]["dist_to_true_end"] for r in rows])), 4)} if n else None,
        "mean_dist_under_cf_instr": {"to_true_end": round(float(np.mean([r["cf_instr_run"]["dist_to_true_end"] for r in rows])), 4),
                                     "to_cf_end": round(float(np.mean([r["cf_instr_run"]["dist_to_cf_end"] for r in rows])), 4)} if n else None,
        "honest_note": ("delivery proxy is Euclidean eef-to-object distance over a real receding-horizon "
                        "rollout, NOT LIBERO's own bddl containment predicate (which only recognizes the "
                        "BDDL-designated target); a 'closer' result is suggestive evidence of behavioral "
                        "grounding, not a certified task-success claim."),
        "rows": rows,
    }
    with open(f"{VOL_PATH}/libero_causal_intervention_closedloop_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=7200)
def libero_gram_ablation_closedloop(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                                    res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                                    norm: str = "rational", max_task: int = 6, eps_per_task: int = 8,
                                    k: int = 64, n_gram_eval: int = 1024,
                                    max_steps: int = 280, num_steps_wait: int = 10):
    """NEXT_PHASE_PLAN.md Part 1, item 1.1 (cleanest alternate intervention): turn the ALREADY-
    POSITIVE offline gripper Gram result (odt_libero_action, up to 42x global-vs-random faithfulness)
    into a REAL behavioral causal test via closed-loop activation patching -- no language touched at
    all, so it completely sidesteps the scene/instruction confound that explains the honest-negative
    instruction-swap test. Builds the gripper-group Gram at the visual-patch bond (same construction
    as odt_libero_action), then runs closed-loop rollouts 3 ways with MATCHED seeds: (i) no
    truncation (baseline, should reproduce ~ the known capability number), (ii) global top-k
    projection, (iii) a random k-subspace of the same size. Does offline faithfulness (measured by
    MSE reconstruction) actually predict closed-loop robustness?"""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.odt import random_projector

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    D = cfg.dim
    print(f"loaded {ckpt}: D={D} d_a={d_a}")

    # ---- build the gripper-group Gram at the visual-patch bond (same construction as odt_libero_action) ----
    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_gram_eval, len(samples)), replace=False)
    imgs_g = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr_g = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states_g = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    emb0_g = torch.zeros(len(idx), dtype=torch.long, device=dev)

    def build_seq_g(vis, sl):
        B = vis.shape[0]
        bos = model.bos.expand(B, -1, -1)
        instr_e = model.tok_emb(instr_g[sl])
        st = model.state_proj(states_g[sl])[:, None]
        embe = model.embodiment_emb(emb0_g[sl])[:, None]
        aq = model.action_queries.expand(B, -1, -1)
        x = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
        return x + model.pos_emb[:, :x.shape[1]]

    def from_vis_g(vis_in, sl, P=None):
        v = vis_in @ P.T if P is not None else vis_in
        x = build_seq_g(v, sl)
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        h = model.backbone(x, mask=mask)
        h = model.norm_out(h)
        return model.action_head(h[:, -H:])

    with torch.no_grad():
        vis_fixed = model._visual_tokens(imgs_g)
    G = torch.zeros(D, D, device=dev, dtype=torch.float64); n_rows = 0
    for i in range(0, vis_fixed.shape[0], 256):
        sl = slice(i, i + 256)
        vb = vis_fixed[sl].detach().requires_grad_(True)
        act = from_vis_g(vb, sl)
        sel = act[:, :, -1].sum()                                    # gripper dim
        gsel, = torch.autograd.grad(sel, vb)
        g = gsel.reshape(-1, D).double()
        G += g.T @ g; n_rows += g.shape[0]
    G = G / n_rows
    ev, V = torch.linalg.eigh(G)
    V_global = V.flip(1)[:, :k].float()
    rng_t = torch.Generator(device=dev).manual_seed(0)
    V_random = random_projector(D, k, dev, rng_t).float()
    P_global = (V_global @ V_global.T)
    P_random = (V_random @ V_random.T)
    print(f"built gripper Gram + top-{k} global/random projectors")

    # ---- closed-loop rollout, 3 conditions, matched seeds ----
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    @torch.no_grad()
    def predict_chunk(obs, instr_ids, P):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        vis = model._visual_tokens(im)
        if P is not None:
            vis = vis @ P.T
        bos = model.bos.expand(1, -1, -1)
        instr_e = model.tok_emb(instr_ids)
        stt = model.state_proj(st)[:, None]
        embe = model.embodiment_emb(torch.zeros(1, dtype=torch.long, device=dev))[:, None]
        aq = model.action_queries.expand(1, -1, -1)
        x = torch.cat([vis, bos, instr_e, stt, embe, aq], dim=1)
        x = x + model.pos_emb[:, :x.shape[1]]
        mask = causal_mask(x.shape[1], device=dev, dtype=x.dtype)
        h = model.norm_out(model.backbone(x, mask=mask))
        a = model.action_head(h[:, -H:])
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()

    def encode_t(s):
        return torch.tensor([encode(s)], device=dev)

    n_tasks = min(max_task, suite.n_tasks)
    conditions = {"none": None, "global_topk": P_global, "random_topk": P_random}
    per_condition = {c: {} for c in conditions}
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        instr_ids = encode_t(task.language)
        init_states = _libero_init_states_or_raise(suite, ti)
        if len(init_states) < eps_per_task:
            raise RuntimeError(
                f"Task {ti} has only {len(init_states)} canonical states for {eps_per_task} trials"
            )
        for cname, P in conditions.items():
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            succ = 0
            for ep in range(eps_per_task):
                env.seed(ti * 100 + ep); obs = env.reset()
                obs = env.set_init_state(init_states[ep])
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
                ok = False; t = 0
                while t < max_steps and not ok:
                    chunk = predict_chunk(obs, instr_ids, P)
                    for kk in range(min(8, H)):
                        a = chunk[kk].copy(); a[-1] = 1.0 if a[-1] > 0 else -1.0
                        obs, r, done, info = env.step(a.tolist()); t += 1
                        if r > 0: ok = True
                        if done or ok or t >= max_steps: break
                succ += int(ok)
            env.close()
            per_condition[cname][task.language] = round(succ / eps_per_task, 3)
            print(f"[{cname} task {ti}] {task.language}: {succ}/{eps_per_task}")

    overall = {c: round(float(np.mean(list(v.values()))), 3) for c, v in per_condition.items()}
    result = {"ckpt": ckpt, "k": k, "max_task": n_tasks, "eps_per_task": eps_per_task,
              "max_steps": max_steps, "num_steps_wait": num_steps_wait,
              "canonical_init_states": True,
              "overall_by_condition": overall, "per_task_by_condition": per_condition,
              "note": ("no truncation should reproduce ~ the known capability number; global-topk "
                       "vs random-topk at matched k tests whether OFFLINE gripper-Gram faithfulness "
                       "(up to 42x, odt_libero_action) predicts CLOSED-LOOP robustness -- a real "
                       "behavioral activation-patching test, no language touched, sidesteps the "
                       "scene/instruction confound entirely.")}
    with open(f"{VOL_PATH}/libero_gram_ablation_closedloop_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=4 * 3600)
def extended_abstract_chi_random_control(
    checkpoint_seed: int = 0,
    draw: int = 3,
    condition: str = "both",
):
    """Run one exact, preregistered coordinated-surgery Haar control on Modal.

    This calls the same fixed Athena entry point used for the first three controls. The
    separate process prevents that script's deliberate monkeypatches from leaking into a
    warm Modal worker.
    """
    import json
    import os
    import subprocess

    if checkpoint_seed not in (0, 1, 2):
        raise ValueError("checkpoint_seed must be 0, 1, or 2")
    if draw not in (3, 4):
        raise ValueError("draw must be 3 or 4")
    condition_args = {
        "both": "keep_random_normmatched,remove_random_normmatched",
        "keep": "keep_random_normmatched",
        "remove": "remove_random_normmatched",
    }
    if condition not in condition_args:
        raise ValueError("condition must be 'both', 'keep', or 'remove'")

    checkpoints = {
        0: "ckpt_linear_rat_vit_s0_v2.pt",
        1: "ckpt_linear_rat_vit_s1.pt",
        2: "ckpt_linear_rat_vit_s2.pt",
    }
    experiment_seed = 2026083200 + 10 * checkpoint_seed + draw
    condition_suffix = "" if condition == "both" else f"_{condition}"
    output_name = (
        f"extended_abstract_chi_surgery_v1_s{checkpoint_seed}_random_d{draw}"
        f"_modal{condition_suffix}.json"
    )
    output_path = os.path.join(VOL_PATH, output_name)
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite completed control: {output_path}")

    def make_command(task_start, task_end, path, surgery_conditions):
        return [
            sys.executable,
            "athena/run_coordinated_surgery_v1.py",
            "--mode", "surgery",
            "--architecture", "chi",
            "--vision-encoder", "vit",
            "--suite", "libero_object",
            "--training-suite", "libero_object",
            "--cache", os.path.join(VOL_PATH, "libero_frames_100000_64.pkl"),
            "--task-start", str(task_start),
            "--task-end", str(task_end),
            "--max-steps", "280",
            "--matmul-precision", "highest",
            "--block-index", "0",
            "--rank", "128",
            "--checkpoint", os.path.join(VOL_PATH, checkpoints[checkpoint_seed]),
            "--output", path,
            "--seed", str(experiment_seed),
            "--eps-per-task", "20",
            "--surgery-conditions", surgery_conditions,
        ]

    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = os.pathsep.join(
        item for item in (PROJ, subprocess_env.get("PYTHONPATH", "")) if item
    )
    subprocess_env["MUJOCO_GL"] = "osmesa"
    subprocess_env["PYOPENGL_PLATFORM"] = "osmesa"
    subprocess_env.pop("MUJOCO_EGL_DEVICE_ID", None)

    command = make_command(4, 10, output_path, condition_args[condition])
    subprocess.run(command, cwd=PROJ, env=subprocess_env, check=True)
    vol.commit()
    with open(output_path) as handle:
        result = json.load(handle)
    summary = {
        "output": output_name,
        "checkpoint_seed": checkpoint_seed,
        "draw": draw,
        "condition": condition,
        "experiment_seed": experiment_seed,
        "overall_by_condition": result["surgery"]["overall_by_condition"],
    }
    print("RESULT:", json.dumps(summary, indent=2))
    return summary


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=4 * 3600)
def libero_attention_weight_surgery(
    ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt",
    n_frames: int = 100000,
    res: int = 64,
    horizon: int = 8,
    block_idx: int = 6,
    k: int = 128,
    max_task: int = 4,
    eps_per_task: int = 5,
    max_steps: int = 280,
    exec_h: int = 8,
    num_steps_wait: int = 10,
    seed: int = 731,
    tag: str = "_screen",
):
    """Weight-only discovery followed by direct coefficient surgery.

    The exact reduced attention Gram is built from the trained Q/K/V/O coefficients only.
    For every head in one preregistered block, its eigenvectors define input-bond projectors.
    We then edit the actual Q1/K1/Q2/K2/V weight columns, with no activation hook, fitting,
    gradient, rollout-derived ranking, or retraining.

    Seven paired closed-loop conditions test two predictions made before evaluation:

    1. Keeping the top-k weight-derived subspace should preserve more success than keeping a
       matched random k-subspace and than a random coefficient edit of equal Frobenius norm.
    2. Removing the top-k subspace should damage success more than removing a matched random
       k-subspace and than a random coefficient edit of equal Frobenius norm.

    This is a coefficient-level causal-interface screen. It is not yet a semantic repair.
    A positive screen justifies using the same interface for a preregistered gripper-timing or
    instruction-shortcut edit.
    """
    _bootstrap()
    import json
    import os
    import pickle
    from collections import defaultdict

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.exact_odt_attention_proto import build_gram_reduced_head

    dev = "cuda"
    H = horizon
    dataset_name = "lerobot/libero_object_image"

    tasks = {}
    task_files = [
        f for f in list_repo_files(dataset_name, repo_type="dataset")
        if "task" in f.lower() and f.endswith((".jsonl", ".json", ".parquet"))
    ]
    for task_file in task_files:
        try:
            task_path = hf_hub_download(dataset_name, task_file, repo_type="dataset")
            if task_path.endswith(".parquet"):
                import pandas as pd
                frame = pd.read_parquet(task_path).reset_index()
                text_col = "task" if "task" in frame.columns else next(
                    c for c in frame.columns if frame[c].dtype == object
                )
                index_col = (
                    "task_index" if "task_index" in frame.columns
                    else ("index" if "index" in frame.columns else frame.columns[0])
                )
                for _, row in frame.iterrows():
                    tasks[int(row[index_col])] = str(row[text_col])
            else:
                for line in open(task_path):
                    row = json.loads(line)
                    tasks[int(row["task_index"])] = row["task"]
            if tasks:
                break
        except Exception as exc:
            print(f"{task_file}: {exc}")

    words = set()
    for task_text in tasks.values():
        words.update(task_text.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for word in sorted(words):
        vocab[word] = len(vocab)
    instruction_len = 32

    def encode(text):
        ids = [1] + [vocab.get(word, 0) for word in text.lower().replace(".", "").split()]
        return (ids[:instruction_len] + [0] * max(0, instruction_len - len(ids)))[:instruction_len]

    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    episodes = defaultdict(list)
    for frame in frames:
        episodes[frame[0]].append(frame)
    samples = []
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda row: row[1])
        for idx in range(len(episode_frames) - H):
            action_chunk = np.stack([episode_frames[idx + step][4] for step in range(H)])
            samples.append(
                (
                    episode_frames[idx][2],
                    episode_frames[idx][5],
                    episode_frames[idx][3],
                    action_chunk,
                )
            )

    action_dim = samples[0][3].shape[1]
    state_dim = samples[0][2].shape[0]
    all_actions = np.stack([sample[3] for sample in samples])
    all_states = np.stack([sample[2] for sample in samples])
    action_mean = all_actions.mean((0, 1))
    action_std = all_actions.std((0, 1)) + 1e-6
    state_mean = all_states.mean(0)
    state_std = all_states.std(0) + 1e-6

    cfg = VLAConfig(
        image_size=res,
        patch_size=8,
        vit_dim=192,
        vit_layers=4,
        vit_heads=8,
        vocab_size=len(vocab),
        max_instr_len=instruction_len,
        state_dim=state_dim,
        n_embodiments=1,
        dim=384,
        n_layers=8,
        n_heads=12,
        action_horizon=H,
        action_dim=action_dim,
        action_head="linear",
        norm="rational",
        qk_norm="rational",
        vision_encoder="vit",
    )
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()

    if not 0 <= block_idx < cfg.n_layers:
        raise ValueError(f"block_idx={block_idx} outside [0,{cfg.n_layers})")
    if not 0 < k < cfg.dim:
        raise ValueError(f"k={k} must be between 1 and {cfg.dim - 1}")

    attention = model.backbone.blocks[block_idx].attn
    if not hasattr(attention, "wq1"):
        raise ValueError("Weight surgery requires the deployed bilinear attention module.")
    head_dim = attention.head_dim
    dim = cfg.dim
    matrix_names = ("wq1", "wk1", "wq2", "wk2", "wv")
    original_weights = {
        name: getattr(attention, name).weight.detach().clone()
        for name in matrix_names
    }

    def weight_and_bias(linear):
        return (
            linear.weight.detach().double().cpu().numpy(),
            linear.bias.detach().double().cpu().numpy(),
        )

    Wq1, bq1 = weight_and_bias(attention.wq1)
    Wk1, bk1 = weight_and_bias(attention.wk1)
    Wq2, bq2 = weight_and_bias(attention.wq2)
    Wk2, bk2 = weight_and_bias(attention.wk2)
    Wv, bv = weight_and_bias(attention.wv)
    Wo = attention.wo.weight.detach().double().cpu().numpy()

    rng = np.random.default_rng(seed)
    projectors = {
        "keep_top": [],
        "keep_random": [],
        "remove_top": [],
        "remove_random": [],
    }
    norm_match_scales = {
        "keep_random_normmatched": [],
        "remove_random_normmatched": [],
    }
    spectra = {}
    for head_idx in range(attention.n_heads):
        row_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
        gram = build_gram_reduced_head(
            Wq1[row_slice],
            bq1[row_slice],
            Wk1[row_slice],
            bk1[row_slice],
            Wq2[row_slice],
            bq2[row_slice],
            Wk2[row_slice],
            bk2[row_slice],
            Wv[row_slice],
            bv[row_slice],
            Wo[:, row_slice],
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        top_basis = eigenvectors[:, :k]
        top_projector = top_basis @ top_basis.T
        random_basis, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        random_projector = random_basis @ random_basis.T
        identity = np.eye(dim)
        projectors["keep_top"].append(top_projector)
        projectors["keep_random"].append(random_projector)
        projectors["remove_top"].append(identity - top_projector)
        projectors["remove_random"].append(identity - random_projector)
        keep_top_delta_sq = 0.0
        keep_random_delta_sq = 0.0
        remove_top_delta_sq = 0.0
        remove_random_delta_sq = 0.0
        for matrix in (Wq1, Wk1, Wq2, Wk2, Wv):
            matrix_slice = matrix[row_slice]
            keep_top_delta_sq += float(
                np.square(matrix_slice @ top_projector - matrix_slice).sum()
            )
            keep_random_delta_sq += float(
                np.square(matrix_slice @ random_projector - matrix_slice).sum()
            )
            remove_top_delta_sq += float(
                np.square(matrix_slice @ (identity - top_projector) - matrix_slice).sum()
            )
            remove_random_delta_sq += float(
                np.square(matrix_slice @ (identity - random_projector) - matrix_slice).sum()
            )
        norm_match_scales["keep_random_normmatched"].append(
            (keep_top_delta_sq / max(keep_random_delta_sq, 1e-30)) ** 0.5
        )
        norm_match_scales["remove_random_normmatched"].append(
            (remove_top_delta_sq / max(remove_random_delta_sq, 1e-30)) ** 0.5
        )
        spectra[head_idx] = {
            "top_eigenvalues": eigenvalues[:8].tolist(),
            "top_k_spectral_mass": float(
                eigenvalues[:k].clip(min=0).sum()
                / max(eigenvalues.clip(min=0).sum(), 1e-30)
            ),
        }

    def restore_weights():
        with torch.no_grad():
            for name in matrix_names:
                getattr(attention, name).weight.copy_(original_weights[name])

    def apply_condition(condition):
        restore_weights()
        if condition == "baseline":
            return 0.0
        squared_delta = 0.0
        squared_base = 0.0
        projector_condition = condition
        if condition == "keep_random_normmatched":
            projector_condition = "keep_random"
        elif condition == "remove_random_normmatched":
            projector_condition = "remove_random"
        with torch.no_grad():
            for head_idx in range(attention.n_heads):
                row_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
                projector = torch.tensor(
                    projectors[projector_condition][head_idx],
                    device=dev,
                    dtype=original_weights["wq1"].dtype,
                )
                interpolation = (
                    norm_match_scales[condition][head_idx]
                    if condition in norm_match_scales
                    else 1.0
                )
                for name in matrix_names:
                    original_slice = original_weights[name][row_slice]
                    projected_slice = original_slice @ projector
                    edited_slice = original_slice + interpolation * (
                        projected_slice - original_slice
                    )
                    getattr(attention, name).weight[row_slice].copy_(edited_slice)
                    squared_delta += float((edited_slice - original_slice).float().pow(2).sum())
                    squared_base += float(original_slice.float().pow(2).sum())
        return (squared_delta / max(squared_base, 1e-30)) ** 0.5

    action_mean_t = torch.tensor(action_mean, device=dev)
    action_std_t = torch.tensor(action_std, device=dev)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32)
    state_std_t = torch.tensor(state_std, dtype=torch.float32)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def build_state(observation):
        raw_state = np.concatenate(
            [
                observation["robot0_eef_pos"],
                quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        if len(raw_state) >= state_dim:
            return raw_state[:state_dim]
        return np.pad(raw_state, (0, state_dim - len(raw_state)))

    @torch.no_grad()
    def predict_chunk(observation, instruction_ids):
        image = np.asarray(
            Image.fromarray(
                np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
            ).resize((res, res))
        )
        image_t = (
            torch.tensor(image)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(dev)
        )
        state_t = (
            (torch.tensor(build_state(observation)) - state_mean_t) / state_std_t
        ).float().unsqueeze(0).to(dev)
        action, _ = model(
            image_t,
            instruction_ids,
            state_t,
            torch.zeros(1, dtype=torch.long, device=dev),
        )
        return (action[0] * action_std_t + action_mean_t).cpu().numpy()

    conditions = (
        "baseline",
        "keep_top",
        "keep_random",
        "keep_random_normmatched",
        "remove_top",
        "remove_random",
        "remove_random_normmatched",
    )
    episode_bits = {condition: {} for condition in conditions}
    relative_weight_change = {}
    n_tasks = min(max_task, suite.n_tasks)

    for condition in conditions:
        relative_weight_change[condition] = apply_condition(condition)
        print(
            f"[{condition}] relative edited-weight Frobenius change "
            f"{relative_weight_change[condition]:.6f}"
        )
        for task_idx in range(n_tasks):
            task = suite.get_task(task_idx)
            bddl_path = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            init_states = _libero_init_states_or_raise(suite, task_idx)
            instruction_ids = torch.tensor([encode(task.language)], device=dev)
            environment = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=res,
                camera_widths=res,
            )
            bits = []
            for episode_idx in range(eps_per_task):
                environment.seed(task_idx * 100 + episode_idx)
                observation = environment.reset()
                observation = environment.set_init_state(
                    init_states[episode_idx % len(init_states)]
                )
                for _ in range(num_steps_wait):
                    observation, _, _, _ = environment.step(dummy_action)
                success = False
                step_count = 0
                while step_count < max_steps and not success:
                    action_chunk = predict_chunk(observation, instruction_ids)
                    for chunk_idx in range(min(exec_h, H)):
                        action = action_chunk[chunk_idx].copy()
                        action[-1] = 1.0 if action[-1] > 0 else -1.0
                        observation, reward, done, _ = environment.step(action.tolist())
                        step_count += 1
                        success = reward > 0
                        if done or success or step_count >= max_steps:
                            break
                bits.append(bool(success))
            environment.close()
            episode_bits[condition][task_idx] = bits
            print(
                f"[{condition} task {task_idx}] {task.language}: "
                f"{sum(bits)}/{len(bits)}"
            )

    restore_weights()
    overall = {
        condition: float(
            np.mean(
                [
                    success
                    for task_bits in episode_bits[condition].values()
                    for success in task_bits
                ]
            )
        )
        for condition in conditions
    }
    paired_predictions = {
        "keep_top_minus_keep_random": overall["keep_top"] - overall["keep_random"],
        "keep_top_minus_keep_random_normmatched": (
            overall["keep_top"] - overall["keep_random_normmatched"]
        ),
        "remove_random_minus_remove_top": overall["remove_random"] - overall["remove_top"],
        "remove_random_normmatched_minus_remove_top": (
            overall["remove_random_normmatched"] - overall["remove_top"]
        ),
    }
    result = {
        "ckpt": ckpt,
        "block_idx": block_idx,
        "k": k,
        "rank_fraction": k / dim,
        "n_heads": attention.n_heads,
        "max_task": n_tasks,
        "eps_per_task": eps_per_task,
        "max_steps": max_steps,
        "conditions": list(conditions),
        "overall_by_condition": overall,
        "per_episode_success": episode_bits,
        "relative_weight_change": relative_weight_change,
        "paired_predictions": paired_predictions,
        "spectra": spectra,
        "preregistered_predictions": {
            "P1": "keep_top success > keep_random success",
            "P2": "remove_top success < remove_random success",
            "P3": "keep_top success > keep_random_normmatched success",
            "P4": "remove_top success < remove_random_normmatched success",
        },
        "discovery_inputs": "trained weights only",
        "semantic_scope": (
            "This screen tests the coefficient interface and predicted behavioral "
            "faithfulness. It does not attach a human semantic label or claim repair."
        ),
    }
    output_path = (
        f"{VOL_PATH}/libero_attention_weight_surgery_b{block_idx}_k{k}{tag}.json"
    )
    with open(output_path, "w") as output_file:
        json.dump(result, output_file, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "spectra"}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=10800)
def libero_closedloop_capability(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                                 res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                                 norm: str = "rational", max_task: int = 10, eps_per_task: int = 20,
                                 max_steps: int = 280, exec_h: int = 8, tag: str = "",
                                 architecture: str = "chi", attn: str = "bilinear",
                                 ffn: str = "bilinear", ffn_rank: int = 0,
                                 vit_ffn_rank: int = 0, num_steps_wait: int = 10):
    """Dedicated, cheap capability-ONLY closed-loop eval -- the single canonical protocol to use
    for both the pre-finetune baseline and every variant of the decorrelated-demo capability-
    collapse fix sweep (DEVLOG cont.62 task #15/#16, RESEARCH_PLAN.md). Every run in that sweep
    MUST call this exact function with identical (max_task, eps_per_task, max_steps) so numbers
    are directly comparable -- this exists specifically to prevent a repeat of cont.62's
    correction #2 (a 600-step-cap headline number silently compared against a 280-step-cap
    replication and reported as a same-protocol check). Rollout mechanics match
    libero_gram_ablation_closedloop's 'none' condition (real receding-horizon predict_chunk,
    exec_h-step chunk execution, gripper sign hard-thresholded at 0, LIBERO's own bddl reward
    r>0 as success) minus the Gram-matrix/autograd pass that function only needs to build its
    (here, irrelevant) truncation projectors -- pure capability signal, nothing else computed."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"; H = horizon
    if architecture == "conventional":
        if vision_encoder != "vit":
            raise ValueError("The controlled conventional twin requires vision_encoder='vit'.")
        attn, ffn = "softmax", "swiglu"
        norm, qk_norm = "per_token", "none"
        ffn_rank = ffn_rank or 1408
        vit_ffn_rank = vit_ffn_rank or 704
    elif architecture == "chi":
        qk_norm = norm
    else:
        raise ValueError("architecture must be 'chi' or 'conventional'")
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=qk_norm,
                    vision_encoder=vision_encoder, attn=attn, ffn=ffn,
                    ffn_rank=ffn_rank or None, vit_ffn_rank=vit_ffn_rank or None)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    print(f"loaded {ckpt}: D={cfg.dim} d_a={d_a}")

    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    @torch.no_grad()
    def predict_chunk(obs, instr_ids):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()

    n_tasks = min(max_task, suite.n_tasks)
    per_task = {}
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        if len(init_states) < eps_per_task:
            raise RuntimeError(
                f"Task {ti} has only {len(init_states)} canonical states for {eps_per_task} trials"
            )
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        succ = 0
        for ep in range(eps_per_task):
            env.seed(ti * 100 + ep); obs = env.reset()
            obs = env.set_init_state(init_states[ep])
            for _ in range(num_steps_wait):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
            ok = False; t = 0
            while t < max_steps and not ok:
                chunk = predict_chunk(obs, instr_ids)
                for kk in range(min(exec_h, H)):
                    a = chunk[kk].copy(); a[-1] = 1.0 if a[-1] > 0 else -1.0
                    obs, r, done, info = env.step(a.tolist()); t += 1
                    if r > 0: ok = True
                    if done or ok or t >= max_steps: break
            succ += int(ok)
        env.close()
        per_task[task.language] = round(succ / eps_per_task, 3)
        print(f"[task {ti}] {task.language}: {succ}/{eps_per_task}")

    overall = round(float(np.mean(list(per_task.values()))), 4)
    n_trials = n_tasks * eps_per_task
    result = {"ckpt": ckpt, "max_task": n_tasks, "eps_per_task": eps_per_task, "max_steps": max_steps,
              "exec_h": exec_h, "n_trials": n_trials, "overall_success_rate": overall,
              "n_success": round(overall * n_trials), "per_task": per_task,
              "architecture": architecture, "attention": attn, "ffn": ffn,
              "norm": norm, "qk_norm": qk_norm, "params": sum(p.numel() for p in model.parameters()),
              "canonical_init_states": True, "num_steps_wait": num_steps_wait,
              "protocol_note": ("canonical capability-only protocol for the decorrelated-demo fix "
                                "sweep -- use identical max_task/eps_per_task/max_steps for the "
                                "baseline (unfinetuned) checkpoint and every swept variant, always "
                                "in the same batch, never comparing against a differently-capped "
                                "number from a different session (cont.62 correction #2).")}
    out = f"{VOL_PATH}/libero_closedloop_capability_{ckpt.replace('.pt', '')}{tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))
    print("per_task:", json.dumps(per_task, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3600)
def odt_depth_resolved_trace(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                             res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                             norm: str = "rational", n_eval: int = 1024, ks=(8, 32)):
    """NEXT_PHASE_PLAN.md Part 1, item 1.2: generalize odt_libero_action's single visual-patch
    bond into a DEPTH-RESOLVED circuit trace -- the same output-sensitivity Gram / global-vs-
    random-k faithfulness / spatial-coherence measurement, but now at all 9 depths: depth 0 (the
    raw visual-patch bond, pre-backbone, == odt_libero_action) and depths 1..8 (the residual
    stream's visual-patch slice immediately after each of the 8 ChiTransformerBlocks). At each
    depth d we take the (detached) residual activation, mark ONLY its visual-patch slice as the
    differentiable leaf (mirrors odt_libero_action's vis_in.requires_grad_(True) exactly, just
    moved deeper), continue the forward pass through blocks[d:] + norm_out + action_head, and
    autograd back to that leaf -- giving a Gram at each depth with the SAME downstream causal
    chain length shrinking as d grows (depth 8 = right before norm_out, zero downstream blocks).
    Answers: does the causally-load-bearing subspace stay ~fixed through depth (a stable circuit)
    or rotate/expand (depth-distributed computation)? Does spatial coherence sharpen or diffuse
    with depth (do patches literally start "voting" for objects, or does info globalize)?"""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.odt import random_projector

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    D = cfg.dim; n_layers = cfg.n_layers
    print(f"loaded {ckpt}: D={D} d_a={d_a} n_layers={n_layers}")

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions_raw = np.stack([samples[i][3] for i in idx])
    Atrue = torch.tensor((actions_raw - a_mu) / a_sd, dtype=torch.float32, device=dev)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    with torch.no_grad():
        vis_fixed = model._visual_tokens(imgs)                        # (B, Nv, D)
        Nv = vis_fixed.shape[1]
        bos = model.bos.expand(len(idx), -1, -1)
        instr_e = model.tok_emb(instr)
        st = model.state_proj(states)[:, None]
        embe = model.embodiment_emb(emb0)[:, None]
        aq = model.action_queries.expand(len(idx), -1, -1)
        x0 = torch.cat([vis_fixed, bos, instr_e, st, embe, aq], dim=1)
        x0 = x0 + model.pos_emb[:, :x0.shape[1]]
        mask = causal_mask(x0.shape[1], device=dev, dtype=x0.dtype)
        hs = [x0.detach()]
        h = x0
        for block in model.backbone.blocks:
            h = block(h, mask=mask)
            hs.append(h.detach())
    print(f"cached {len(hs)} depth activations, Nv={Nv}, seq_len={x0.shape[1]}")

    def continue_from(h_full, d):
        h = h_full
        for block in model.backbone.blocks[d:]:
            h = block(h, mask=mask)
        h = model.norm_out(h)
        return model.action_head(h[:, -H:])                            # (B,H,d_a)

    groups = {"eef_transl": list(range(min(3, d_a))), "rotation": list(range(3, min(6, d_a))),
              "gripper": [d_a - 1]}
    rng = torch.Generator(device=dev).manual_seed(0)

    def gram_at_depth(d, dims):
        h_d = hs[d]
        G = torch.zeros(D, D, device=dev, dtype=torch.float64); n = 0
        for i in range(0, h_d.shape[0], 256):
            sl = slice(i, i + 256)
            vis_leaf = h_d[sl, :Nv].detach().requires_grad_(True)
            rest = h_d[sl, Nv:].detach()
            h_full = torch.cat([vis_leaf, rest], dim=1)
            act = continue_from(h_full, d)
            sel = act[:, :, dims].sum()
            gsel, = torch.autograd.grad(sel, vis_leaf)
            g = gsel.reshape(-1, D).double()
            G += g.T @ g; n += g.shape[0]
        return G / n

    def topk_proj(G, k):
        ev, V = torch.linalg.eigh(G)
        return V.flip(1)[:, :k]

    @torch.no_grad()
    def group_mse_at_depth(d, dims, P):
        h_d = hs[d]; tot = n = 0.0
        for i in range(0, h_d.shape[0], 256):
            sl = slice(i, i + 256)
            v = h_d[sl, :Nv]
            v = v @ P.T if P is not None else v
            rest = h_d[sl, Nv:]
            h_full = torch.cat([v, rest], dim=1)
            pred = continue_from(h_full, d)
            tot += (pred[:, :, dims] - Atrue[sl][:, :, dims]).pow(2).sum().item()
            n += pred[:, :, dims].numel()
        return round(tot / n, 5)

    @torch.no_grad()
    def patch_locality(h_d, v):
        proj = torch.einsum("bnd,d->bn", h_d[:, :Nv].double(), v.double())
        energy = (proj ** 2).mean(0)
        k_top = max(1, int(round(0.1 * Nv)))
        return (torch.topk(energy, k_top).values.sum() / energy.sum().clamp_min(1e-30)).item()

    rng_loc = torch.Generator(device=dev).manual_seed(0)
    depth_results = {}
    for d in range(n_layers + 1):
        faith_d, coh_d = {}, {}
        for gname, dims in groups.items():
            if not dims: continue
            G = gram_at_depth(d, dims)
            full_mse = group_mse_at_depth(d, dims, None)
            Vg = topk_proj(G, max(ks))
            curve_g, curve_r = {}, {}
            for k in ks:
                Pg = (Vg[:, :k] @ Vg[:, :k].T).float()
                curve_g[k] = group_mse_at_depth(d, dims, Pg)
                curve_r[k] = group_mse_at_depth(d, dims, random_projector(D, k, dev, rng).float())
            faith_d[gname] = {"full_mse": full_mse, "global": curve_g, "random": curve_r}
            v_top = topk_proj(G, 1)[:, 0]
            rand_locs = [patch_locality(hs[d], torch.randn(D, generator=rng_loc, device=dev)) for _ in range(20)]
            coh_d[gname] = {"top_causal_dir_locality": round(patch_locality(hs[d], v_top), 4),
                            "random_dir_locality_mean": round(sum(rand_locs) / len(rand_locs), 4)}
        depth_results[d] = {"faithfulness_by_group": faith_d, "coherence_by_group": coh_d}
        print(f"[depth {d}] done:", json.dumps(depth_results[d], indent=2))

    result = {"ckpt": ckpt, "norm": norm, "vision_encoder": vision_encoder, "D": D, "d_a": d_a,
              "n_layers": n_layers, "n_eval": len(idx), "ks": list(ks), "by_depth": depth_results,
              "note": ("depth 0 == odt_libero_action's bond exactly (raw visual-patch tokens, "
                       "pre-backbone); depths 1..n_layers are the residual stream's visual-patch "
                       "slice immediately after each ChiTransformerBlock. Downstream causal chain "
                       "shrinks as depth grows (depth n_layers has zero blocks left, only "
                       "norm_out+action_head) -- expect global/random faithfulness ratio to fall "
                       "toward 1 as depth increases purely from that shrinking-chain effect; the "
                       "interesting signal is whether it falls FASTER or SLOWER than that trivial "
                       "expectation, and whether spatial coherence sharpens or diffuses with depth.")}
    with open(f"{VOL_PATH}/odt_depth_resolved_trace_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT SUMMARY:", json.dumps({d: r["faithfulness_by_group"] for d, r in depth_results.items()}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3600)
def odt_attention_real_weights(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                                res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                                norm: str = "rational", n_eval: int = 1024,
                                block_idxs="0,6,7", ks="8,32,128", n_rand_dirs: int = 8,
                                tag: str = ""):
    """block_idxs/ks: comma-separated STRINGS, not tuples (Modal's CLI has no reliable way to pass
    a list to an 'ANY'-typed function param -- repeated `--block-idxs N` flags silently keep only
    the LAST value rather than accumulating, a real mistake this session made and caught, see
    DEVLOG cont.67: a first "8-block" run silently only ran block 7). Parsed via .split(",") below.

    DEVLOG cont.65 / user task: extract REAL LEARNED Q/K/V/O weights (+ rn_q1/rn_k1/rn_q2/rn_k2's
    REAL fitted running_ms) for one attention head at real depth from the actual trained checkpoint
    (dim=384, n_heads=12, head_dim=32, qk_norm='rational'), build the EXACT weight-only Gram via
    the closed-form O(dim^2) reduction validated in xvla/train/exact_odt_attention_proto.py Sec 7
    (2.13e-14 max err vs brute-force order-6 tensor at toy scale -- see that file's `run_sec7()`),
    and compare its top-k-vs-random-k faithfulness curve on REAL activations (not synthetic
    Gaussian noise) against the toy proto's modest 1.0-2.5x result on RANDOM UNTRAINED weights.

    WHY 3 BLOCKS, NOT 1. odt_depth_resolved_trace found the gripper causal ratio peaks at "depth
    7" (hs[7] = residual state after blocks[0..6], i.e. immediately BEFORE block index 7, the
    last block) -- but that Gram is built from the RESIDUAL STREAM at that depth, not from one
    attention layer's weights, so which single block index "corresponds" to the peak is genuinely
    ambiguous (block 6 produced it; block 7 consumes it). Rather than guess, sweep block 0
    (baseline/first), block 6 (produces the peak-depth representation), and block 7 (last, closest
    to the action head) and report all three -- exactly this project's own established discipline
    of not cherry-picking one depth (NEXT_PHASE_PLAN.md Sec 1.2).

    WEIGHT EXTRACTION, PRECISELY. wq1/wk1/wq2/wk2/wv are nn.Linear(dim,dim); _project() reshapes
    each to (B,H,N,head_dim) and slices head h as OUTPUT rows [h*head_dim:(h+1)*head_dim] of the
    FULL dim-wide matrix -- so one head's Wq1_h etc is a (head_dim, dim) ROW-slice (all `dim` input
    columns kept: the per-head projection still reads the WHOLE 384-dim token, this is the "dim
    mismatch" the task asked about -- head_dim shrinks the OUTPUT/bottleneck, not the input/
    homogeneous dimension D1=dim+1=385, which is why naively reusing the toy's build_core_tensors
    at this scale would need a (384,385,385,385,385,385) tensor -- 3.2e15 floats, infeasible; see
    exact_odt_attention_proto.py Sec 7's docstring for the full derivation of why the closed-form
    reduction sidesteps this). Wo is column-sliced to [: , h*head_dim:(h+1)*head_dim] (all `dim`
    output rows kept). bo is EXCLUDED from any single head's core (bias isn't separable across
    heads -- it's added once, after summing all heads). attn_gain (ChiTransformerBlock's residual
    scale, `x = x + attn_gain*attn(u)`) is NOT part of BilinearAttention.forward() itself -- so it
    is excluded when verifying against the real module directly (Check B below), and reported
    SEPARATELY as a second "residual-stream-facing" Gram (G_stream = attn_gain**2 * G_raw, exact
    since G is quadratic in Wo and attn_gain multiplies Wo's effective contribution linearly).
    """
    block_idxs = [int(b) for b in str(block_idxs).split(",")]
    ks = [int(k) for k in str(ks).split(",")]
    _bootstrap()
    import copy, json, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from xvla.train.exact_odt_attention_proto import build_gram_reduced_head

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    D, n_heads, n_layers = cfg.dim, cfg.n_heads, cfg.n_layers
    head_dim = D // n_heads
    print(f"loaded {ckpt}: D={D} n_heads={n_heads} head_dim={head_dim} n_layers={n_layers} qk_norm={norm}")
    assert norm == "rational", "this function's rational-norm handling assumes qk_norm=norm='rational'"

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    with torch.no_grad():
        vis_fixed = model._visual_tokens(imgs)
        bos = model.bos.expand(len(idx), -1, -1)
        instr_e = model.tok_emb(instr)
        st = model.state_proj(states)[:, None]
        embe = model.embodiment_emb(emb0)[:, None]
        aq = model.action_queries.expand(len(idx), -1, -1)
        x0 = torch.cat([vis_fixed, bos, instr_e, st, embe, aq], dim=1)
        x0 = x0 + model.pos_emb[:, :x0.shape[1]]
        mask = causal_mask(x0.shape[1], device=dev, dtype=x0.dtype)
        hs = [x0.detach()]
        h = x0
        for block in model.backbone.blocks:
            h = block(h, mask=mask)
            hs.append(h.detach())
    Nseq = x0.shape[1]
    print(f"cached {len(hs)} depth activations, seq_len={Nseq}, n_eval={len(idx)}")

    def extract_block(block_idx):
        """Real weights + real rational-norm state for every head of one block, as float64 numpy."""
        attn = model.backbone.blocks[block_idx].attn
        def w(lin): return lin.weight.detach().double().cpu().numpy()
        def b(lin): return lin.bias.detach().double().cpu().numpy()
        Wq1, Wk1, Wq2, Wk2, Wv, Wo = w(attn.wq1), w(attn.wk1), w(attn.wq2), w(attn.wk2), w(attn.wv), w(attn.wo)
        bq1, bk1, bq2, bk2, bv, bo = b(attn.wq1), b(attn.wk1), b(attn.wq2), b(attn.wk2), b(attn.wv), b(attn.wo)
        attn_gain = float(model.backbone.blocks[block_idx].attn_gain.detach().cpu())
        rn = {}
        for tag in ("q1", "k1", "q2", "k2"):
            m = getattr(attn, f"rn_{tag}")
            rn[tag] = dict(s0=float(m.running_ms.detach().cpu()),
                           pa=m.pa.detach().double().cpu().numpy(),
                           pb=m.pb.detach().double().cpu().numpy(), deg=m.deg)
        return dict(Wq1=Wq1, bq1=bq1, Wk1=Wk1, bk1=bk1, Wq2=Wq2, bq2=bq2, Wk2=Wk2, bk2=bk2,
                    Wv=Wv, bv=bv, Wo=Wo, bo=bo, attn_gain=attn_gain, rn=rn)

    def rat_norm_real(x, rn_entry):
        """Sec 6's rational_norm() generalizes as-is: ms is computed over the LAST axis (here
        head_dim), so calling it per-head with the REAL pa/pb/s0 (not the toy's placeholder
        0.8/1.1/0.9/1.05) reproduces RationalNorm(pade).forward() exactly for that head."""
        ms = (x ** 2).mean(axis=-1, keepdims=True) + 1e-6
        v = ms / rn_entry["s0"]
        P = sum(rn_entry["pa"][k] * v ** k for k in range(rn_entry["deg"] + 1))
        Q = sum(rn_entry["pb"][k] * v ** k for k in range(rn_entry["deg"] + 1))
        r = (P / Q) * (rn_entry["s0"] ** -0.5)
        return x * r

    def direct_forward_all_heads_rational(X, W):
        """Exact reconstruction of attn.forward(X, method='explicit') (qk_norm='rational'),
        summed over all n_heads (NO attn_gain -- that's the block wrapper's, not this module's).
        X: (N, D). Matches direct_forward_rational's math (Sec 6) but multi-head."""
        N = X.shape[0]
        out = np.zeros((N, D))
        m = np.tril(np.ones((N, N)))
        n_i = np.maximum(m.sum(-1), 1.0)
        c = (1.0 / np.sqrt(n_i))[:, None]
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            q1 = X @ W["Wq1"][sl].T + W["bq1"][sl]; k1 = X @ W["Wk1"][sl].T + W["bk1"][sl]
            q2 = X @ W["Wq2"][sl].T + W["bq2"][sl]; k2 = X @ W["Wk2"][sl].T + W["bk2"][sl]
            v = X @ W["Wv"][sl].T + W["bv"][sl]
            q1n, k1n = rat_norm_real(q1, W["rn"]["q1"]), rat_norm_real(k1, W["rn"]["k1"])
            q2n, k2n = rat_norm_real(q2, W["rn"]["q2"]), rat_norm_real(k2, W["rn"]["k2"])
            a = (q1n @ k1n.T) * (q2n @ k2n.T) / (head_dim ** 2)
            a = a * m
            y = c * (a @ v)
            out += y @ W["Wo"][:, sl].T
        return out + W["bo"]

    results = {}
    for block_idx in block_idxs:
        W = extract_block(block_idx)
        u_real = model.backbone.blocks[block_idx].rbn_attn(hs[block_idx]).double().cpu().numpy()

        # ---- Check A (cheap sanity): sliced weights reproduce the real per-head affine maps. ----
        with torch.no_grad():
            u_t = model.backbone.blocks[block_idx].rbn_attn(hs[block_idx])
            q1_real_all = (u_t @ model.backbone.blocks[block_idx].attn.wq1.weight.T
                          + model.backbone.blocks[block_idx].attn.wq1.bias)
            q1_real_all = q1_real_all.double().cpu().numpy()
        h0 = slice(0, head_dim)
        q1_mine_h0 = u_real[0] @ W["Wq1"][h0].T + W["bq1"][h0]
        err_slice = float(np.max(np.abs(q1_real_all[0, :, h0] - q1_mine_h0)))

        # ---- Check B (load-bearing): full multi-head real-rational-norm reconstruction vs the
        # REAL torch module's own forward(method='explicit'), on REAL activations, float64.
        # MUST deepcopy+.double() the submodule (not just cast the input) -- nn.Linear rejects a
        # float64 input against float32 weights. Expected floor ~1e-7/1e-8, NOT 1e-14, because
        # RationalNorm.forward() hardcodes `xf = x.float()` internally regardless of caller dtype
        # (xvla/nn/normalization.py:275, already documented in odt_terminal_bond_atlas / Sec 6) --
        # this is the SAME already-diagnosed float32 floor, not a new bug if it recurs here. ----
        with torch.no_grad():
            attn64 = copy.deepcopy(model.backbone.blocks[block_idx].attn).double()
            real_out = attn64(u_t.double(), mask=mask.double(), method="explicit")[0].cpu().numpy()
        mine_out = direct_forward_all_heads_rational(u_real[0], W)
        err_full = float(np.max(np.abs(real_out - mine_out)))
        print(f"[block {block_idx}] check A (per-head slice) max err {err_slice:.3e} | "
              f"check B (full multi-head reconstruction vs real module) max err {err_full:.3e}")

        # ---- Per-head closed-form Gram (build_gram_reduced_head, O(dim^2), Sec 7). The GRAM
        # itself is built from the qk_norm='none'-equivalent weight-only core (matching the toy
        # proto's Sec 5/Sec 6 division of labor exactly: Kfull is UNCHANGED by rational norm,
        # which only ever contributes extra per-token scalar factors at EVALUATION time) -- but
        # the CURVE evaluation below (full/top-k/random-k) uses the RATIONAL-NORM-AWARE per-head
        # forward (`head_forward_rational`, mirroring the toy's `factored_forward_rational`), NOT
        # the plain biquadratic `direct_forward_head` -- using the norm-free evaluator here would
        # silently test a different function (qk_norm='none') than the one actually deployed. ----
        def head_forward_rational(X, sl, rn):
            q1 = X @ W["Wq1"][sl].T + W["bq1"][sl]; k1 = X @ W["Wk1"][sl].T + W["bk1"][sl]
            q2 = X @ W["Wq2"][sl].T + W["bq2"][sl]; k2 = X @ W["Wk2"][sl].T + W["bk2"][sl]
            v = X @ W["Wv"][sl].T + W["bv"][sl]
            q1n, k1n = rat_norm_real(q1, rn["q1"]), rat_norm_real(k1, rn["k1"])
            q2n, k2n = rat_norm_real(q2, rn["q2"]), rat_norm_real(k2, rn["k2"])
            a = (q1n @ k1n.T) * (q2n @ k2n.T) / (head_dim ** 2)
            N = X.shape[0]
            m = np.tril(np.ones((N, N)))
            a = a * m
            c = (1.0 / np.sqrt(np.maximum(m.sum(-1), 1.0)))[:, None]
            y = c * (a @ v)
            return y @ W["Wo"][:, sl].T

        per_head = {}
        i0 = Nseq - 1   # last token (sees every key under the causal mask)
        Xreal = u_real   # (n_eval, Nseq, D) -- real, not synthetic Gaussian
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            args_raw = (W["Wq1"][sl], W["bq1"][sl], W["Wk1"][sl], W["bk1"][sl],
                        W["Wq2"][sl], W["bq2"][sl], W["Wk2"][sl], W["bk2"][sl],
                        W["Wv"][sl], W["bv"][sl], W["Wo"][:, sl])
            G_raw = build_gram_reduced_head(*args_raw)
            G_stream = (W["attn_gain"] ** 2) * G_raw
            evals = np.linalg.eigvalsh(G_stream)[::-1]
            evecs = np.linalg.eigh(G_stream)[1][:, ::-1]

            curve = {}
            for k in ks:
                Vk = evecs[:, :k]; Ptop = Vk @ Vk.T
                mse_top, mse_rand = 0.0, 0.0
                n_test = min(200, Xreal.shape[0])
                rr = np.random.default_rng(1000 * block_idx + h)
                # Draw the matched random subspaces once and reuse them for every held-out
                # activation.  Re-drawing and QR-factorizing inside the activation loop made
                # the original full eight-block sweep need nearly one million large QRs.  A
                # fixed control bank is both the cleaner paired comparison and over 100x
                # cheaper, while retaining n_rand_dirs independent random controls.
                random_projectors = []
                for _ in range(n_rand_dirs):
                    A_ = rr.normal(size=(D, k))
                    Vr, _ = np.linalg.qr(A_)
                    random_projectors.append(Vr @ Vr.T)
                for b_i in rr.choice(Xreal.shape[0], size=n_test, replace=False):
                    X = Xreal[b_i]                                   # (Nseq, D), REAL activations
                    full = head_forward_rational(X, sl, W["rn"])[i0]
                    Xt = X.copy(); Xt[:i0 + 1] = Xt[:i0 + 1] @ Ptop.T
                    top = head_forward_rational(Xt, sl, W["rn"])[i0]
                    mse_top += np.mean((full - top) ** 2)
                    rand_acc = 0.0
                    for Pr in random_projectors:
                        Xr = X.copy(); Xr[:i0 + 1] = Xr[:i0 + 1] @ Pr.T
                        rand_out = head_forward_rational(Xr, sl, W["rn"])[i0]
                        rand_acc += np.mean((full - rand_out) ** 2)
                    mse_rand += rand_acc / n_rand_dirs
                mse_top /= n_test; mse_rand /= n_test
                curve[k] = dict(mse_top=mse_top, mse_rand=mse_rand,
                                ratio=(mse_rand / mse_top if mse_top > 0 else float("inf")))
            per_head[h] = dict(top_eigvals=evals[:8].tolist(), curve=curve)

        results[block_idx] = dict(check_A_slice_err=err_slice, check_B_full_reconstruction_err=err_full,
                                  attn_gain=W["attn_gain"],
                                  rn_running_ms={t: W["rn"][t]["s0"] for t in W["rn"]},
                                  per_head=per_head)

    out = {"ckpt": ckpt, "D": D, "n_heads": n_heads, "head_dim": head_dim, "n_layers": n_layers,
          "block_idxs": list(block_idxs), "ks": list(ks), "n_eval": len(idx), "by_block": results,
          "note": ("check_A/check_B ~1e-8-1e-13 confirms the extracted affine slices and the "
                   "separate rational-aware replay are faithful to the real module (bounded by "
                   "RationalNorm's internal float32 cast, same floor "
                   "documented in odt_terminal_bond_atlas/Sec 6 -- NOT a derivation bug if err ~1e-7"
                   "-1e-8 rather than ~1e-14). The reduced Gram is exact only for the polynomial "
                   "affine Q/K/V/O core before input-dependent rational Q/K rescaling. "
                   "curve[k].ratio > 1 and rising with k means the core-derived top-k subspace is "
                   "more predictive of this head's real rational output than a random "
                   "same-size subspace, evaluated on REAL LIBERO activations (not synthetic Gaussian"
                   " like the toy proto) -- the direct test of whether real learned weights give a "
                   "stronger faithfulness curve than the toy's untrained-weights 1.0-2.5x.")}
    with open(
        f"{VOL_PATH}/odt_attention_real_weights_{ckpt.replace('.pt','')}{tag}.json",
        "w",
    ) as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({b: {"check_A": r["check_A_slice_err"], "check_B": r["check_B_full_reconstruction_err"],
                                     "ratios_k_max": {h: per_head["curve"][max(ks)]["ratio"]
                                                     for h, per_head in r["per_head"].items()}}
                                 for b, r in results.items()}, indent=2))
    return out


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def odt_sae_hybrid(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                   res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                   norm: str = "rational", n_eval: int = 512, k_gram: int = 64,
                   n_components: int = 16, top_n: int = 6, group: str = "gripper"):
    """NEXT_PHASE_PLAN.md Part 1, item 1.3: ODT-then-SAE hybrid, not SAE-on-raw-activations.
    Several very recent (2025-26) papers fit sparse dictionaries on raw VLA activations to find
    nameable features on black-box models with no causal basis to start from. Here we fit a cheap
    FastICA rotation INSIDE the already causally-ranked top-k_gram Gram subspace at the visual-
    patch bond (same Gram construction as odt_libero_action) for one action group at a time --
    every resulting component inherits a global importance guarantee for free (it lives, by
    construction, in a subspace already shown to carry the group's causal signal); the open
    question becomes purely "what does this mean", answered two ways: (1) top-activating image-
    patch lookup (crops saved to the volume, the standard feature-dashboard method), (2) causal-
    ablation replay -- zero one component (in the causal subspace, then project back to the full
    D-dim bond) and remeasure ALL THREE action-group MSEs, so we can see whether an atom is
    SELECTIVE (only its own group's error moves) or bleeds into unrelated groups."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask
    from sklearn.decomposition import FastICA
    from PIL import Image as PILImage

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    D = cfg.dim
    print(f"loaded {ckpt}: D={D} d_a={d_a} n_samples={len(samples)}")

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    actions_raw = np.stack([samples[i][3] for i in idx])
    Atrue = torch.tensor((actions_raw - a_mu) / a_sd, dtype=torch.float32, device=dev)
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    def build_seq(vis, sl):
        B = vis.shape[0]
        bos = model.bos.expand(B, -1, -1)
        instr_e = model.tok_emb(instr[sl])
        st = model.state_proj(states[sl])[:, None]
        embe = model.embodiment_emb(emb0[sl])[:, None]
        aq = model.action_queries.expand(B, -1, -1)
        x = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
        return x + model.pos_emb[:, :x.shape[1]]

    def from_vis(vis_in, sl):
        x = build_seq(vis_in, sl)
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        h = model.backbone(x, mask=mask)
        h = model.norm_out(h)
        return model.action_head(h[:, -H:])

    with torch.no_grad():
        vis_fixed = model._visual_tokens(imgs)                       # (B, Nv, D)
    Nv = vis_fixed.shape[1]
    grid = int(round(Nv ** 0.5))
    groups = {"eef_transl": list(range(min(3, d_a))), "rotation": list(range(3, min(6, d_a))),
              "gripper": [d_a - 1]}
    dims = groups[group]

    # ---- (1) the causal Gram for this group at the visual-patch bond (same construction as odt_libero_action) ----
    G = torch.zeros(D, D, device=dev, dtype=torch.float64); n_rows = 0
    for i in range(0, vis_fixed.shape[0], 256):
        sl = slice(i, i + 256)
        vb = vis_fixed[sl].detach().requires_grad_(True)
        act = from_vis(vb, sl)
        sel = act[:, :, dims].sum()
        gsel, = torch.autograd.grad(sel, vb)
        g = gsel.reshape(-1, D).double()
        G += g.T @ g; n_rows += g.shape[0]
    G = G / n_rows
    ev, V = torch.linalg.eigh(G)
    V_k = V.flip(1)[:, :k_gram]                                        # (D, k_gram), THE causal subspace
    print(f"built '{group}' Gram, top-{k_gram} causal subspace")

    # ---- (2) FastICA rotation *inside* that subspace, on per-patch features ----
    with torch.no_grad():
        Z = (vis_fixed.double() @ V_k).float()                         # (B, Nv, k_gram)
    Zflat = Z.reshape(-1, k_gram).cpu().numpy()
    ica = FastICA(n_components=n_components, random_state=0, max_iter=2000, whiten="unit-variance")
    Scomp = ica.fit_transform(Zflat)                                    # (B*Nv, n_components)
    S_r = Scomp.reshape(len(idx), Nv, n_components)
    print(f"fit FastICA: {n_components} components inside the top-{k_gram} '{group}' subspace")

    # ---- (2b) reconstruction baseline with ALL n_components kept (none zeroed). Ablating one
    # atom must be compared against THIS, not the true full-D model -- fitting n_components < D
    # is itself lossy (a rank-64 causal-Gram projection further compressed to n_components
    # independent axes), so comparing an ablated atom directly against the unprojected full
    # model would conflate "this one atom's marginal effect" with "the whole k_gram/ICA
    # compression's lossiness", which is common to every atom and swamps the per-atom signal.
    s_full = ica.transform(Zflat)
    z_rec_full = ica.inverse_transform(s_full)
    v_rec_full = torch.tensor(z_rec_full, device=dev, dtype=torch.float64).reshape(len(idx), Nv, k_gram).float()
    vis_recon_full = (v_rec_full.double() @ V_k.T).float()

    @torch.no_grad()
    def group_mse_of(vis_tensor, dims2):
        tot = n_tot = 0.0
        for i in range(0, vis_fixed.shape[0], 256):
            sl = slice(i, i + 256)
            pred = from_vis(vis_tensor[sl], sl)
            tot += (pred[:, :, dims2] - Atrue[sl][:, :, dims2]).pow(2).sum().item()
            n_tot += pred[:, :, dims2].numel()
        return tot / n_tot

    true_full_mse = {gname2: round(group_mse_of(vis_fixed, dims2), 5) for gname2, dims2 in groups.items() if dims2}
    recon_all_mse = {gname2: round(group_mse_of(vis_recon_full, dims2), 5) for gname2, dims2 in groups.items() if dims2}
    print(f"true full-model MSE (unprojected): {true_full_mse}")
    print(f"reconstruction-with-all-{n_components}-components-kept MSE (the correct ablation baseline): {recon_all_mse}")

    # ---- (3) label each component: top-activating patch crops ----
    os.makedirs(f"{VOL_PATH}/sae_crops_{group}", exist_ok=True)
    atom_reports = []
    for c in range(n_components):
        act_c = S_r[:, :, c]
        flat = np.argsort(-np.abs(act_c).ravel())[:top_n]
        examples = []
        for rank, fi in enumerate(flat):
            b, p = np.unravel_index(fi, act_c.shape)
            row, col = p // grid, p % grid
            examples.append({"sample": int(b), "patch_row": int(row), "patch_col": int(col),
                             "activation": round(float(act_c[b, p]), 4),
                             "task": tasks.get(samples[idx[b]][1], "")})
            ps = res // grid
            crop = (imgs[b, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps]
                    .clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            PILImage.fromarray(crop).resize((64, 64), PILImage.NEAREST).save(
                f"{VOL_PATH}/sae_crops_{group}/atom{c:02d}_rank{rank}.png")

        # ---- (4) causal ablation replay: zero JUST this component (all other n_components-1
        # components kept), remeasure ALL group MSEs against the recon_all_mse baseline (same
        # k_gram/ICA compression, just this one atom removed) -- isolates THIS atom's marginal
        # effect from the compression's own lossiness, which recon_all_mse already captures.
        s0 = ica.transform(Zflat)
        s0[:, c] = 0.0
        z_rec = ica.inverse_transform(s0)                               # (B*Nv, k_gram)
        v_rec = torch.tensor(z_rec, device=dev, dtype=torch.float64).reshape(len(idx), Nv, k_gram).float()
        vis_ablated = (v_rec.double() @ V_k.T).float()                  # back to D-dim bond, only this atom removed
        deltas = {}
        for gname2, dims2 in groups.items():
            if not dims2: continue
            abl_mse = round(group_mse_of(vis_ablated, dims2), 5)
            base = recon_all_mse[gname2]
            deltas[gname2] = {"recon_all_components_mse": base, "this_atom_ablated_mse": abl_mse,
                              "true_full_model_mse": true_full_mse[gname2],
                              "pct_increase_vs_recon_baseline": round(100 * (abl_mse - base) / max(base, 1e-12), 1)}
        atom_reports.append({"component": c, "top_examples": examples, "causal_ablation_by_group": deltas})
        print(f"[atom {c}] top task words: {[e['task'][:30] for e in examples[:3]]} | ablation deltas: {deltas}")

    result = {"ckpt": ckpt, "group": group, "k_gram": k_gram, "n_components": n_components,
              "n_eval": len(idx), "Nv": Nv, "grid": grid, "atoms": atom_reports,
              "note": ("each atom lives inside the already causally-verified top-k Gram subspace "
                       "for `group`; 'selective' atoms (ablation delta concentrated in `group`, "
                       "near-zero for the other two) are the strongest candidates for a genuinely "
                       "nameable, causally-grounded visual feature; crops saved under "
                       f"sae_crops_{group}/ on the volume for visual inspection.")}
    with open(f"{VOL_PATH}/odt_sae_hybrid_{group}_{vision_encoder}_{norm}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1200)
def odt_terminal_bond_atlas(ckpt: str = "ckpt_linear_rat_vit_s0_v2.pt", n_frames: int = 100000,
                            res: int = 64, horizon: int = 8, vision_encoder: str = "vit",
                            norm: str = "rational", n_eval: int = 4096):
    """NEXT_PHASE_PLAN.md Part 1, item 1: the EXACT, weight-only, data-free terminal-bond atlas.
    Zero forward-pass approximation, zero training. Key fact: action_head is nn.Linear, and
    norm_out (RationalNorm) is DIRECTION-PRESERVING -- it multiplies the whole activation vector
    by one positive per-token SCALAR (both 'pade' and 'meansq' variants: inv_sqrt/ms are shape
    (...,1), broadcast over the feature dim), so norm_out(x) = g(x)*x, g(x)>0, never mixing
    coordinates. That means w_a = action_head.weight[a] (unit-normalized) IS, exactly, the
    direction the deployed model reads for action dim a at the pre-norm_out bond -- no Gram,
    no data-driven approximation needed. This becomes the free ground-truth anchor every
    data-driven Gram elsewhere (odt_libero_action) can be checked against."""
    _bootstrap()
    import json, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.attention import causal_mask

    dev = "cuda"; H = horizon
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt}", map_location=dev, weights_only=True))
    model.eval()
    print(f"loaded {ckpt}: dim={cfg.dim} d_a={d_a} n_samples={len(samples)}")

    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(samples), size=min(n_eval, len(samples)), replace=False)
    imgs = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255).to(dev)
    instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idx], device=dev)
    states = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    A_raw = np.stack([samples[i][3] for i in idx])                       # (B,H,d_a) raw units
    emb0 = torch.zeros(len(idx), dtype=torch.long, device=dev)

    with torch.no_grad():
        vis = model._visual_tokens(imgs)
        bos = model.bos.expand(len(idx), -1, -1)
        instr_e = model.tok_emb(instr)
        st = model.state_proj(states)[:, None]
        embe = model.embodiment_emb(emb0)[:, None]
        aq = model.action_queries.expand(len(idx), -1, -1)
        x0 = torch.cat([vis, bos, instr_e, st, embe, aq], dim=1)
        x0 = x0 + model.pos_emb[:, :x0.shape[1]]
        mask = causal_mask(x0.shape[1], device=dev, dtype=x0.dtype)
        h_prenorm = model.backbone(x0, mask=mask)[:, -H:]                  # (B,H,dim) pre-norm_out
        h_postnorm = model.norm_out(h_prenorm)                             # (B,H,dim)

    # ---- (1) MECHANICALLY VERIFY direction-preservation: cos(norm_out(x), x) == 1 ----
    hp = h_prenorm.reshape(-1, cfg.dim).double()
    hn = h_postnorm.reshape(-1, cfg.dim).double()
    cos = (hp * hn).sum(-1) / (hp.norm(dim=-1) * hn.norm(dim=-1) + 1e-12)
    direction_preserving = {
        "min_cos": round(float(cos.min()), 10), "mean_cos": round(float(cos.mean()), 10),
        "max_abs_1_minus_cos": round(float((1 - cos).abs().max()), 12),
    }
    print("DIRECTION-PRESERVING CHECK:", json.dumps(direction_preserving, indent=2))

    # ---- (2) THE EXACT ATLAS: w_a = action_head.weight[a], unit-normalized ----
    W = model.action_head.weight.detach().double()               # (d_a, dim)
    Wn = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)        # unit rows, THE exact atlas

    # ---- (3) sanity: does w_a . h_prenorm actually predict the real action, end to end? ----
    a_sd_t = torch.tensor(a_sd, device=dev).double(); a_mu_t = torch.tensor(a_mu, device=dev).double()
    with torch.no_grad():
        pred_norm, _ = model(imgs, instr, states, emb0)                    # (B,H,d_a) normalized-space pred
    pred_raw = (pred_norm.double() * a_sd_t + a_mu_t).reshape(-1, d_a)      # exact model output, raw units
    target_raw = torch.tensor(A_raw.reshape(-1, d_a), device=dev).double()
    exact_read = hp @ Wn.T                                                  # (rows, d_a): w_a . h_prenorm, unit-w_a
    bias = model.action_head.bias.detach().double()
    # NOTE: ChiVLA.forward applies norm_out to the FULL sequence BEFORE slicing to aq_out, so the
    # action_head actually reads the POST-norm bond (hn), not hp. Reconstruct from hn for the
    # true bookkeeping check; hp is used above only for the direction (w_a), which is valid at
    # either bond since norm_out is direction-preserving (confirmed by the cos==1 check above).
    exact_full = hn @ W.T + bias                                            # should equal pred_norm exactly (pre-denorm)
    recon_err = float((exact_full - pred_norm.double().reshape(-1, d_a)).abs().max())

    per_dim = {}
    for a in range(d_a):
        r_norm = float(np.corrcoef(exact_read[:, a].cpu().numpy(), target_raw[:, a].cpu().numpy())[0, 1])
        per_dim[a] = {"corr_examined_direction_vs_real_action": round(r_norm, 4)}

    result = {"ckpt": ckpt, "d_a": d_a, "dim": cfg.dim, "n_eval": len(idx),
              "direction_preserving_check": direction_preserving,
              "exact_read_reconstructs_model_output_max_abs_err": round(recon_err, 12),
              "per_action_dim_atlas_vs_real_action_corr": per_dim,
              "note": ("w_a=action_head.weight[a] is an EXACT weight-only direction (zero forward-pass "
                       "approximation) since norm_out never mixes coordinates; every data-driven Gram at "
                       "earlier bonds (odt_libero_action) can be checked for subspace overlap against this.")}
    with open(f"{VOL_PATH}/odt_terminal_bond_atlas_{vision_encoder}_{norm}.json", "w") as f:
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


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=2 * 3600)
def libero_action_multimodality(n_frames: int = 20000, res: int = 64, horizon: int = 8,
                                n_anchors: int = 2000, k: int = 40, n_img_pc: int = 32,
                                state_weight: float = 1.0, gap_tau: float = 2.0,
                                mass_tau: float = 0.2, per_task: bool = True,
                                source: str = "lerobot", success_only: bool = True):
    """EMPIRICAL test of CONDITIONAL action MULTIMODALITY in the LIBERO demos.

    Question (DEVLOG cont.17): given (near-)identical observations, is the distribution
    of next-action-chunks unimodal or multimodal? If near-unimodal, LIBERO-Object is the
    WRONG task to demonstrate a multimodal head beating a mean head (the MSE conditional
    mean is already near-optimal → explains linear 25% > flow/product 0%).

    Method (CPU, no GPU, reads the frame cache):
      1. Observation embedding e_i = [ z(PCA(gray 16x16 image)) | state_weight * z(state) ].
         (Same obs the policy conditions on: image + proprio; language is constant per task.)
      2. For each of n_anchors sampled frames, find its k nearest neighbours in obs space
         *from DIFFERENT episodes* (independent rollouts through a near-identical state — the
         only way to observe conditional multimodality; same-episode neighbours are trivially
         autocorrelated). This is the empirical conditional P(A | O ≈ o_i).
      3. Action chunk a_i = per-dim-standardized next-H actions (H*d_a), split into ARM
         (dims 0..d_a-2) and GRIPPER (dim d_a-1) — the gripper open/close is the obvious
         nuisance bimodality; the arm is the claim under test.
      4. Per neighbourhood, three complementary multimodality statistics:
         (a) DISPERSION: within-neighbourhood action std / global std. ≪1 ⇒ action is
             tightly determined by the obs (necessary, not sufficient, for unimodality).
         (b) BIMODALITY (gap test, dip-test cousin): project the neighbourhood's chunks onto
             the GLOBAL leading arm action-PC; fit optimal 1-D 2-means; standardized
             gap = |μ2-μ1|/pooled_within_std and mass balance min(n1,n2)/n. A neighbourhood
             is "bimodal" iff gap>gap_tau AND mass>mass_tau AND permutation p<0.05 vs a
             Gaussian (unimodal) null of the SAME size (gap is shift/scale-invariant so the
             null depends only on k+1 → computed once). Reports bimodal fraction + dip-style
             bimodality coefficient BC=(skew^2+1)/kurtosis (>0.555 ⇒ non-unimodal).
         (c) INVALID-MEAN (the closed-loop-relevant one): d_mean = ||mean_chunk −
             nearest_real_chunk|| / (median within-neighbourhood pairwise dist). If the
             conditional MEAN action is itself a valid demo action (d_mean small) the linear
             MSE head is safe; if the mean falls in an empty valley between modes (d_mean≫1)
             averaging produces an invalid middle → a mean head MUST fail and a multimodal
             head is needed. This is exactly the failure the closed-loop measures.

    VERDICT:
      NEAR-UNIMODAL (confirms cont.17, LIBERO-Object is the wrong demo task) if
        median dispersion_ratio < ~0.35  AND  arm bimodal_frac < ~0.10  AND
        invalid_mean p90 (arm) < ~1.0 (mean is a valid action almost everywhere).
      GENUINELY MULTIMODAL (refutes; re-open the flow/product 0% as NOT task-mismatch) if
        arm bimodal_frac > ~0.25 with well-separated modes AND invalid_mean median > ~1.5.
    """
    _bootstrap()
    import os, pickle, json
    import numpy as np
    from collections import defaultdict

    if source.startswith("openvla_"):
        # Teacher trajectories (openvla_collect layout): (ti, ep, t, img64, instr, action7, state8, done_ok).
        # Remap to the lerobot frame layout this fn expects: (episode, step, img, state, action, task).
        # Measures the multimodality of EXACTLY the data we distilled the student on.
        tag = source.split("openvla_", 1)[1]
        raw = pickle.load(open(f"{VOL_PATH}/openvla_traj_{tag}.pkl", "rb"))
        if success_only:
            raw = [f for f in raw if f[7]]
        frames = [(int(f[0]) * 1000 + int(f[1]), int(f[2]), f[3],
                   np.asarray(f[6], dtype=np.float32), np.asarray(f[5], dtype=np.float32),
                   int(f[0])) for f in raw]
        print(f"{len(frames)} teacher frames from openvla_traj_{tag}.pkl "
              f"(success_only={success_only})")
    else:
        cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
        frames = pickle.load(open(cache, "rb"))
        print(f"{len(frames)} frames from {cache}")

    # ---- group by episode, build per-frame (obs, next-H chunk) aligned arrays ----
    eps = defaultdict(list)
    for f in frames:
        eps[int(f[0])].append(f)
    imgs, states, chunks, epi, task = [], [], [], [], []
    for ep, fs in eps.items():
        fs.sort(key=lambda z: int(z[1]))
        acts = np.stack([f[4] for f in fs]).astype(np.float32)           # (T, d_a)
        for i in range(len(fs) - horizon):
            imgs.append(fs[i][2]); states.append(fs[i][3])
            chunks.append(acts[i:i + horizon].reshape(-1))               # (H*d_a,)
            epi.append(ep); task.append(int(fs[i][5]))
    imgs = np.stack(imgs); states = np.stack(states).astype(np.float32)
    chunks = np.stack(chunks).astype(np.float32)
    epi = np.asarray(epi); task = np.asarray(task)
    N, d_a = len(chunks), states.shape[0] if states.ndim == 1 else None
    d_a = frames[0][4].shape[0]
    print(f"{N} obs/chunk samples, d_a={d_a}, H={horizon}, episodes={len(eps)}")

    # ---- observation embedding: PCA(16x16 gray) + z(state) ----
    gray = imgs.astype(np.float32).mean(-1)                              # (N,res,res)
    b = res // 16
    gray = gray[:, :b * 16, :b * 16].reshape(N, 16, b, 16, b).mean((2, 4)).reshape(N, -1)
    gray = (gray - gray.mean(0)) / (gray.std(0) + 1e-6)
    # PCA via SVD on (mean-centered) gray
    U, S, Vt = np.linalg.svd(gray - gray.mean(0), full_matrices=False)
    img_emb = (gray @ Vt[:n_img_pc].T)
    img_emb = (img_emb - img_emb.mean(0)) / (img_emb.std(0) + 1e-6)
    st_emb = (states - states.mean(0)) / (states.std(0) + 1e-6)
    obs_emb = np.concatenate([img_emb, state_weight * st_emb], axis=1).astype(np.float32)

    # ---- action normalization + global arm PC1 ----
    a_mu, a_sd = chunks.mean(0), chunks.std(0) + 1e-6
    chunks_n = (chunks - a_mu) / a_sd
    arm_cols = np.array([j for j in range(horizon * d_a) if j % d_a != d_a - 1])
    grip_cols = np.array([j for j in range(horizon * d_a) if j % d_a == d_a - 1])
    arm = chunks_n[:, arm_cols]; grip = chunks_n[:, grip_cols]
    Ua, Sa, Vta = np.linalg.svd(arm - arm.mean(0), full_matrices=False)
    arm_pc1 = Vta[0]                                                     # leading arm direction
    global_arm_std = arm.std(0).mean(); global_grip_std = grip.std(0).mean()

    # ---- Gaussian (unimodal) null for the standardized 2-means gap at size k+1 ----
    def two_means_gap(x):
        x = np.sort(x); n = len(x)
        if n < 8: return 0.0, 0.0
        cs = np.cumsum(x); cs2 = np.cumsum(x * x)
        w_best, j_best = None, 1
        for j in range(1, n):
            n1, n2 = j, n - j
            s1 = cs[j - 1]; s2 = cs[-1] - s1
            ss1 = cs2[j - 1] - s1 * s1 / n1
            ss2 = (cs2[-1] - cs2[j - 1]) - s2 * s2 / n2
            w = ss1 + ss2
            if w_best is None or w < w_best: w_best, j_best = w, j
        j = j_best; mu1 = cs[j - 1] / j; mu2 = (cs[-1] - cs[j - 1]) / (n - j)
        pooled = np.sqrt(max(w_best / (n - 2), 1e-12))
        return abs(mu2 - mu1) / (pooled + 1e-9), min(j, n - j) / n

    rng = np.random.default_rng(0)
    null = np.array([two_means_gap(rng.standard_normal(k + 1))[0] for _ in range(2000)])
    def gap_pvalue(g): return float((1 + (null >= g).sum()) / (len(null) + 1))

    def bimodality_coefficient(x):
        x = np.asarray(x); n = len(x); m = x.mean(); s = x.std() + 1e-9
        z = (x - m) / s
        skew = (z ** 3).mean(); kurt = (z ** 4).mean()
        g1 = skew; g2 = kurt - 3.0
        denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-9)
        return float((g1 ** 2 + 1) / (denom + 1e-9))

    # ---- sample anchors, kNN over obs (cross-episode), compute per-neighbourhood stats ----
    anchors = rng.choice(N, size=min(n_anchors, N), replace=False)
    agg = {"arm_disp": [], "grip_disp": [], "arm_gap": [], "arm_mass": [], "arm_p": [],
           "arm_bc": [], "grip_gap": [], "invalid_mean_arm": [], "invalid_mean_full": [],
           "grip_bimodal": []}
    by_task = defaultdict(lambda: {"arm_bimodal": [], "invalid_mean_arm": [], "arm_disp": []})
    CH = 4096
    for ai in anchors:
        d = obs_emb - obs_emb[ai]
        # chunked squared distance to keep memory flat
        dist = np.empty(N, np.float32)
        for s in range(0, N, CH):
            dist[s:s + CH] = (d[s:s + CH] ** 2).sum(1)
        dist[epi == epi[ai]] = np.inf                                   # cross-episode only
        nn = np.argpartition(dist, k)[:k]
        nn = nn[np.isfinite(dist[nn])]
        if len(nn) < max(8, k // 2): continue
        idx = np.concatenate([[ai], nn])
        ca = chunks_n[idx]                                              # (k+1, H*d_a)
        arm_n = ca[:, arm_cols]; grip_n = ca[:, grip_cols]
        agg["arm_disp"].append(float(arm_n.std(0).mean() / (global_arm_std + 1e-9)))
        agg["grip_disp"].append(float(grip_n.std(0).mean() / (global_grip_std + 1e-9)))
        # bimodality on arm PC1
        proj = arm_n @ arm_pc1
        g, mb = two_means_gap(proj)
        agg["arm_gap"].append(g); agg["arm_mass"].append(mb)
        agg["arm_p"].append(gap_pvalue(g)); agg["arm_bc"].append(bimodality_coefficient(proj))
        # gripper bimodality (mean gripper over chunk per neighbour)
        gm = grip_n.mean(1)
        gg, gmb = two_means_gap(gm)
        agg["grip_gap"].append(gg)
        agg["grip_bimodal"].append(int(gg > gap_tau and gmb > mass_tau and gap_pvalue(gg) < 0.05))
        # invalid-mean: dist(mean chunk, nearest real chunk) / median pairwise
        mean_arm = arm_n.mean(0)
        d_to_real = np.linalg.norm(arm_n - mean_arm, axis=1)
        med_pair = np.median(np.linalg.norm(arm_n - arm_n.mean(0), axis=1)) + 1e-9
        agg["invalid_mean_arm"].append(float(d_to_real.min() / med_pair))
        mean_full = ca.mean(0)
        d_full = np.linalg.norm(ca - mean_full, axis=1)
        med_full = np.median(np.linalg.norm(ca - ca.mean(0), axis=1)) + 1e-9
        agg["invalid_mean_full"].append(float(d_full.min() / med_full))
        bimodal = int(g > gap_tau and mb > mass_tau and agg["arm_p"][-1] < 0.05)
        if per_task:
            t = int(task[ai])
            by_task[t]["arm_bimodal"].append(bimodal)
            by_task[t]["invalid_mean_arm"].append(agg["invalid_mean_arm"][-1])
            by_task[t]["arm_disp"].append(agg["arm_disp"][-1])

    def q(a, p): return round(float(np.percentile(a, p)), 3)
    A = {kk: np.asarray(v, float) for kk, v in agg.items() if len(v)}
    arm_bimodal = ((A["arm_gap"] > gap_tau) & (A["arm_mass"] > mass_tau) & (A["arm_p"] < 0.05))
    summary = {
        "n_anchors_used": int(len(A["arm_disp"])), "k": k,
        "dispersion_ratio_arm": {"median": q(A["arm_disp"], 50), "p90": q(A["arm_disp"], 90)},
        "dispersion_ratio_grip": {"median": q(A["grip_disp"], 50), "p90": q(A["grip_disp"], 90)},
        "arm_bimodal_frac": round(float(arm_bimodal.mean()), 3),
        "arm_gap_median": q(A["arm_gap"], 50), "arm_bc_median": q(A["arm_bc"], 50),
        "arm_bc_frac_gt_0.555": round(float((A["arm_bc"] > 0.555).mean()), 3),
        "grip_bimodal_frac": round(float(np.mean(A["grip_bimodal"])), 3),
        "invalid_mean_arm": {"median": q(A["invalid_mean_arm"], 50),
                             "p90": q(A["invalid_mean_arm"], 90)},
        "invalid_mean_full": {"median": q(A["invalid_mean_full"], 50),
                              "p90": q(A["invalid_mean_full"], 90)},
    }
    near_unimodal = (summary["dispersion_ratio_arm"]["median"] < 0.35
                     and summary["arm_bimodal_frac"] < 0.10
                     and summary["invalid_mean_arm"]["p90"] < 1.0)
    summary["verdict"] = ("NEAR-UNIMODAL (mean head suffices; LIBERO-Object is the wrong "
                          "demo for multimodal benefit)" if near_unimodal else
                          "MULTIMODAL SIGNAL PRESENT (re-open flow/product 0% as NOT task-mismatch)")
    if per_task:
        summary["per_task"] = {int(t): {
            "arm_bimodal_frac": round(float(np.mean(d["arm_bimodal"])), 3),
            "invalid_mean_arm_median": round(float(np.median(d["invalid_mean_arm"])), 3),
            "arm_disp_median": round(float(np.median(d["arm_disp"])), 3),
        } for t, d in sorted(by_task.items()) if d["arm_bimodal"]}
    _sfx = "" if source == "lerobot" else f"_{source}"
    with open(f"{VOL_PATH}/libero_action_multimodality{_sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    print("MULTIMODALITY:", json.dumps(summary, indent=2))
    return summary


@app.function(image=image, gpu="A10G", timeout=2 * 3600)
def synth_multimodal_eval(steps: int = 4000, sep: float = 0.5, eps_hit: float = 0.12,
                          flow_steps: int = 20, n_factors: int = 3):
    """DEMONSTRATION task where a multimodal head PROVABLY beats a mean head.

    AMBIGUOUS REACH (xvla/train/synth_vla.make_ambiguous_batch): the scene contains TWO
    identical valid targets (same colour+shape) at symmetric positions p_L, p_R separated
    by `sep`; the instruction ("reach the {colour} {shape}") does NOT disambiguate which.
    Demos are 50/50: each trajectory reaches ONE of the two (a genuine conditional bimodal
    target — the SAME obs+instruction maps to two valid chunks). Everything else matches
    synth_vla (32x32 render, ChiVLA, on-device, no downloads).

    WHY A MEAN HEAD MUST FAIL (provable): the L2-optimal deterministic map given a 50/50
    bimodal target is the MEAN = the midpoint (p_L+p_R)/2, which is the EMPTY SPACE between
    the two objects. With sep > 2*eps_hit the midpoint is > eps_hit from BOTH targets ⇒ the
    linear head's endpoint lands on nothing ⇒ commit_rate = 0 by construction. A multimodal
    head (flow / product) represents both chunks and commits to one valid target.

    METRIC (separates 'commits to a valid mode' from 'averages into an invalid middle'):
      endpoint e = gripper_start + Σ_h step_h (integrate the predicted chunk).
      commit  = min(||e-p_L||, ||e-p_R||) < eps_hit           (reached a real target)
      average = ||e - midpoint|| < eps_hit AND not commit      (stuck in the empty valley)
    Report per head: commit_rate (higher=better, the multimodal win) and average_rate
    (higher=worse, the mean-collapse signature). Expectation: linear commit≈0/average≈1;
    flow & product commit≫0.
    """
    _bootstrap()
    import json
    import numpy as np
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.synth_vla import make_ambiguous_batch, VOCAB
    dev = "cuda"; H = 4

    def train_eval(head):
        cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=128, vit_layers=3, vit_heads=8,
                        vocab_size=VOCAB, max_instr_len=16, state_dim=8, n_embodiments=1,
                        dim=256, n_layers=6, n_heads=8, action_horizon=H, action_dim=7,
                        action_head=head, flow_steps=flow_steps, n_factors=n_factors,
                        # symmetric-bimodal task → flow needs KWTA decode (mode maps to
                        # the empty valley on a symmetric conditional; DEVLOG cont.19 #3).
                        flow_decode="kwta")
        m = ChiVLA(cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.05)
        m.train()
        for step in range(steps):
            batch = make_ambiguous_batch(128, dev, sep=sep)
            opt.zero_grad(set_to_none=True)
            _, loss = m(batch["img"], batch["instr"], batch["state"], batch["embodiment"],
                        target_actions=batch["actions"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % 1000 == 0: print(f"[{head}] step {step} loss {loss.item():.4f}")
        m.eval()
        with torch.no_grad():
            b = make_ambiguous_batch(2048, dev, sep=sep)
            pred, _ = m(b["img"], b["instr"], b["state"], b["embodiment"])   # (B,H,7)
            start = b["state"][:, :2]                                        # gripper start
            e = start + pred[:, :, :2].sum(1)                                # endpoint (x,y)
            pL, pR = b["p_left"], b["p_right"]; mid = (pL + pR) / 2
            dL = (e - pL).norm(dim=1); dR = (e - pR).norm(dim=1)
            dmid = (e - mid).norm(dim=1)
            commit = (torch.minimum(dL, dR) < eps_hit)
            average = (dmid < eps_hit) & (~commit)
        return {"commit_rate": round(float(commit.float().mean()), 3),
                "average_rate": round(float(average.float().mean()), 3),
                "mean_endpoint_err_to_nearest": round(float(torch.minimum(dL, dR).mean()), 3)}

    out = {h: train_eval(h) for h in ["linear", "flow", "product", "quantile"]}
    out["config"] = {"sep": sep, "eps_hit": eps_hit, "steps": steps}
    print("SYNTH MULTIMODAL EVAL:", json.dumps(out, indent=2))
    return out


@app.function(image=libero_image, volumes={VOL_PATH: vol}, timeout=3 * 3600)
def libero_gen_decorrelated_demos(max_task: int = 10, eps_per_task: int = 30, res: int = 64,
                                  max_steps: int = 250, kp: float = 8.0, hover: float = 0.10,
                                  grasp_z_offset: float = 0.0, grasp_dwell: int = 15,
                                  lift: float = 0.15, drop_offset: float = 0.08,
                                  release_dwell: int = 10, phase_timeout: int = 60, seed0: int = 5000,
                                  success_xy_radius: float = 0.12, success_z_max: float = 0.15,
                                  out_pkl: str = "libero_decorrelated_demos.pkl",
                                  out_tasks: str = "libero_decorrelated_tasks.json",
                                  save_debug_frames: bool = False):
    """The real fix for the closed-loop instruction-swap honest negative (DEVLOG cont.57/59):
    every LIBERO-Object scene template is paired with an almost-fixed instruction in the real
    dataset, so BC can solve tasks via scene-identity recognition rather than word->object
    binding (de Haan/Jayaraman/Levine, "Causal Confusion in Imitation Learning", NeurIPS 2019).
    This generates NEW demonstrations that decorrelate scene from instruction: for each (task
    template, episode) we pick ONE object uniformly (round-robin) from ALL present graspable
    objects -- usually a distractor, sometimes the real BDDL target -- and run a scripted
    proportional-control pick-place oracle to it, relabeling the instruction to name whichever
    object was actually reached. CPU-only, no GPU, no model in the loop.

    Reuses already-debugged pieces verbatim: the object-key filter (excludes robot/basket/
    derived '_to_'/'eef' keys) and readable()/instruction-template from
    libero_causal_intervention_closedloop; the gripper sign convention (+1 closes) and
    z-lowering convention from libero_diag's scripted sim test; build_state's state schema.
    Self-verifies success with a generous geometric proxy (final object xy-distance to basket
    center, and z not absurdly high) since LIBERO's own bddl predicate only recognizes the true
    target -- failed episodes are discarded, not kept as bad demos. Output plugs into the exact
    frame-tuple schema build_frame_cache/libero_rollout_head already use, with episode_index
    offset by +1_000_000 and task_index offset into the 1000+ namespace so it can never collide
    with the real 0-9 task indices; the instruction dict is a separate JSON to be merged into
    `tasks` at fine-tune time exactly like other functions here already build `tasks`+`vocab`.

    IMPORTANT: run a small pilot first (max_task=1, eps_per_task=2, save_debug_frames=True) and
    visually inspect the saved PNGs before trusting kp/hover/dwell at scale -- this is a fresh,
    untuned scripted controller, not yet validated on this env."""
    _bootstrap()
    import os, re, json, pickle
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    def build_state(obs):
        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)

    def obj_filter(obs):
        return {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
               if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
               and "_to_" not in k and "eef" not in k}

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = min(max_task, suite.n_tasks)

    all_frames = []          # (episode_index, frame_index, img_uint8, state_f32, action_f32, task_index)
    tasks_out = {}
    ep_idx_base = 1_000_000
    attempted = kept = 0
    debug_dir = f"{VOL_PATH}/decorrelated_debug"
    if save_debug_frames:
        os.makedirs(debug_dir, exist_ok=True)

    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        for ep in range(eps_per_task):
            seed = seed0 + ti * 1000 + ep
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            env.seed(seed); obs = env.reset()
            for _ in range(10):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1.0])

            obj_pos0 = obj_filter(obs)
            basket_key = next((k for k in obs if "basket" in k and k.endswith("_pos")), None)
            if not obj_pos0 or basket_key is None:
                env.close(); continue
            names = sorted(obj_pos0.keys())
            obj_rank = attempted % len(names)
            chosen = names[obj_rank]
            attempted += 1
            grasp_target_z = float(obj_pos0[chosen][2])
            basket_xy = np.asarray(obs[basket_key][:2])
            basket_z = float(obs[basket_key][2])

            task_index = 1000 + ti * 100 + obj_rank
            instr = f"pick up the {readable(chosen)} and place it in the basket"
            tasks_out[task_index] = instr
            episode_index = ep_idx_base + ti * 10000 + ep

            ep_frames = []
            phase = "hover"; phase_t = 0; t = 0
            # BLEND FIX (DEVLOG cont.64/65): libero_decorrelated_demo_diagnostic found the P-control
            # target jumps INSTANTANEOUSLY at every phase transition (mean L2 jump 1.356 on a
            # [-1,1]-clipped scale, ~100% of episodes, almost always at lift->transit) while the
            # end-effector hasn't moved -- nothing like a smooth human teleop trajectory, and a
            # plausible root cause of the capability collapse independent of fine-tune dosage.
            # Fix: linearly blend the EFFECTIVE target from its pre-transition value to the new
            # phase's target over BLEND_WINDOW steps instead of switching instantaneously.
            BLEND_WINDOW = 18
            blend_from = None; blend_step = BLEND_WINDOW   # no prior phase to blend from at t=0
            debug_frames = []
            while t < max_steps:
                obj_pos_t = obj_filter(obs)
                p_obj = np.asarray(obj_pos_t.get(chosen, obj_pos0[chosen]))
                p_eef = np.asarray(obs["robot0_eef_pos"])

                # every distance-gated phase also force-advances after `phase_timeout` steps, so
                # a phase the controller can't quite converge on (steady-state P-control error,
                # a missed grasp) never strands the whole episode -- release is always reached.
                phase_before = phase
                if phase == "hover":
                    target_raw = p_obj + np.array([0.0, 0.0, hover]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "descend", 0
                elif phase == "descend":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.02 or phase_t >= phase_timeout: phase, phase_t = "grasp", 0
                elif phase == "grasp":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = 1.0
                    if phase_t >= grasp_dwell: phase, phase_t = "lift", 0
                elif phase == "lift":
                    target_raw = np.array([p_obj[0], p_obj[1], grasp_target_z + lift]); grip = 1.0
                    if abs(p_eef[2] - (grasp_target_z + lift)) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "transit", 0
                elif phase == "transit":
                    target_raw = np.array([basket_xy[0], basket_xy[1], grasp_target_z + lift]); grip = 1.0
                    if np.linalg.norm(target_raw[:2] - p_eef[:2]) < 0.04 or phase_t >= phase_timeout: phase, phase_t = "drop_descend", 0
                elif phase == "drop_descend":
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = 1.0
                    if abs(p_eef[2] - (basket_z + drop_offset)) < 0.035 or phase_t >= phase_timeout: phase, phase_t = "release", 0
                else:  # release
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = -1.0

                if phase != phase_before:
                    # a transition was just decided (takes effect next iteration) -- start blending
                    # from THIS step's (still pre-transition) target.
                    blend_from = target_raw.copy(); blend_step = 0
                if blend_step < BLEND_WINDOW:
                    alpha = (blend_step + 1) / BLEND_WINDOW
                    target = blend_from + alpha * (target_raw - blend_from)
                    blend_step += 1
                else:
                    target = target_raw

                delta = np.clip(kp * (target - p_eef), -1.0, 1.0)
                action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip], dtype=np.float32)

                img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)), dtype=np.uint8)
                state = build_state(obs)
                ep_frames.append((episode_index, t, img, state, action, task_index))
                if save_debug_frames and t % 10 == 0:
                    debug_frames.append((t, phase, img.copy()))

                obs, r, done, info = env.step(action.tolist())
                phase_t += 1; t += 1
                if phase == "release" and phase_t > release_dwell: break

            final_obj = obj_filter(obs).get(chosen, p_obj)
            final_xy_d = float(np.linalg.norm(np.asarray(final_obj[:2]) - basket_xy))
            final_z = float(final_obj[2])
            success = final_xy_d < success_xy_radius and final_z < success_z_max
            env.close()

            if save_debug_frames:
                sample_idx = np.linspace(0, len(debug_frames) - 1, min(10, len(debug_frames))).astype(int) if debug_frames else []
                montage = np.concatenate([debug_frames[i][2] for i in sample_idx], axis=1) if len(sample_idx) else None
                if montage is not None:
                    Image.fromarray(montage).save(f"{debug_dir}/task{ti}_ep{ep}_{chosen}_{'OK' if success else 'FAIL'}.png")
                sampled_phases = [debug_frames[i][1] for i in sample_idx]
                print(f"[pilot task {ti} ep {ep}] chosen={chosen} success={success} "
                     f"final_xy_d={final_xy_d:.3f} final_z={final_z:.3f} final_phase={phase} "
                     f"n_steps={t} montage_phases={sampled_phases}")

            if success:
                all_frames.extend(ep_frames)
                kept += 1
            print(f"[gen task {ti} ep {ep}] chosen={chosen} instr='{instr}' success={success} "
                 f"final_xy_d={round(final_xy_d,3)} final_z={round(final_z,3)} kept_total={kept}/{attempted}")

    result_summary = {"attempted": attempted, "kept": kept, "kept_frac": round(kept / max(attempted, 1), 3),
                      "n_frames": len(all_frames), "n_tasks_used": n_tasks,
                      "note": ("success = a GENEROUS geometric proxy (final chosen-object xy-distance to "
                               "basket center < success_xy_radius, z < success_z_max), not LIBERO's own bddl "
                               "predicate (which only recognizes the true target). Failed episodes discarded, "
                               "not kept as bad demos.")}
    if not save_debug_frames:
        pickle.dump(all_frames, open(f"{VOL_PATH}/{out_pkl}", "wb"))
        json.dump(tasks_out, open(f"{VOL_PATH}/{out_tasks}", "w"))
        json.dump(result_summary, open(f"{VOL_PATH}/libero_gen_decorrelated_demos_summary.json", "w"), indent=2)
        vol.commit()
    print("RESULT:", json.dumps(result_summary, indent=2))
    return result_summary


@app.function(image=libero_image, volumes={VOL_PATH: vol}, timeout=1800)
def libero_decorrelated_demo_diagnostic(max_task: int = 3, eps_per_task: int = 10, res: int = 64,
                                        max_steps: int = 250, kp: float = 8.0, hover: float = 0.10,
                                        grasp_z_offset: float = 0.0, grasp_dwell: int = 15,
                                        lift: float = 0.15, drop_offset: float = 0.08,
                                        release_dwell: int = 10, phase_timeout: int = 60, seed0: int = 5000,
                                        success_xy_radius: float = 0.12, success_z_max: float = 0.15,
                                        grasp_lift_thresh: float = 0.05, blend: bool = False,
                                        blend_window: int = 18):
    """CPU-only, no-GPU diagnostic for the demo-QUALITY hypothesis behind the decorrelated-demo
    capability collapse (95%->6.5%). Re-runs the exact scripted P-controller from
    libero_gen_decorrelated_demos verbatim (same phase logic, same kp/gains, and now the same
    optional `blend`/`blend_window` phase-transition-smoothing fix, off by default here so
    `blend=False` reproduces the ORIGINAL diagnostic's already-verified numbers exactly, and
    `blend=True` measures whether the fix actually collapses `mean_max_action_jump`) but WITHOUT altering
    it, adding pure instrumentation to directly test three concrete failure modes instead of
    guessing from priors:

    1. SHOVE-NOT-GRASP CONTAMINATION: the generator's success proxy only checks the chosen
       object's FINAL resting xy/z -- it never checks the object was actually lifted off the
       table. A bang-bang P-controller can physically shove/sweep an ungrasped object toward the
       basket and still pass the generous geometric check. We track `max_lift`, the peak
       (object_z(t) - object_z(0)) observed at any point in the episode, and flag
       `success and max_lift < grasp_lift_thresh` as a probable shove: the instruction says "pick
       up X and place it in the basket" but the demonstrated action sequence never picked
       anything up.
    2. BANG-BANG SATURATION: kp=8.0 means any xy/z position error > 1/8=0.125m saturates the
       clipped action at exactly +-1.0. We track `frac_saturated`, the fraction of steps where
       any of the 3 translational action components hits the +-1.0 clip bound exactly -- real
       teleop actions are essentially never at a hard clip boundary for a large fraction of a
       trajectory.
    3. PHASE-TRANSITION DISCONTINUITY: targets jump instantaneously at each of the 6 phase
       transitions (hover->descend->grasp->lift->transit->drop_descend->release) while p_eef has
       not moved yet, so the *commanded action* itself jumps discontinuously at every transition
       (unlike smooth human teleop). We track `max_action_jump`, the largest L2 step-to-step
       change in the 3 translational action components across the whole episode, and where in
       the phase sequence it occurred.

    Reuses libero_gen_decorrelated_demos's exact object/basket selection, phase state machine,
    and success proxy verbatim so the diagnostic is measuring the SAME generator that produced
    the demos already used in train_vla_libero_finetune_decorrelated -- not a reimplementation
    that could disagree with it. Writes per-episode + aggregate stats to the volume; no model,
    no checkpoint, no fine-tuning -- pure sanity check before spending any more GPU-hours."""
    _bootstrap()
    import os, json
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    def build_state(obs):
        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)

    def obj_filter(obs):
        return {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
               if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
               and "_to_" not in k and "eef" not in k}

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = min(max_task, suite.n_tasks)

    episodes = []
    attempted = kept = shoves = 0
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        for ep in range(eps_per_task):
            seed = seed0 + ti * 1000 + ep
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            env.seed(seed); obs = env.reset()
            for _ in range(10):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1.0])

            obj_pos0 = obj_filter(obs)
            basket_key = next((k for k in obs if "basket" in k and k.endswith("_pos")), None)
            if not obj_pos0 or basket_key is None:
                env.close(); continue
            names = sorted(obj_pos0.keys())
            obj_rank = attempted % len(names)
            chosen = names[obj_rank]
            attempted += 1
            obj_z0 = float(obj_pos0[chosen][2])
            grasp_target_z = obj_z0
            basket_xy = np.asarray(obs[basket_key][:2])
            basket_z = float(obs[basket_key][2])

            phase = "hover"; phase_t = 0; t = 0
            max_lift = -1e9
            max_jump = 0.0; jump_phase = None
            n_sat = 0; n_steps = 0
            prev_action_xyz = None
            phase_seq = []
            blend_from = None; blend_step = blend_window   # no prior phase to blend from at t=0
            while t < max_steps:
                obj_pos_t = obj_filter(obs)
                p_obj = np.asarray(obj_pos_t.get(chosen, obj_pos0[chosen]))
                p_eef = np.asarray(obs["robot0_eef_pos"])
                max_lift = max(max_lift, float(p_obj[2]) - obj_z0)

                prev_phase = phase
                if phase == "hover":
                    target_raw = p_obj + np.array([0.0, 0.0, hover]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "descend", 0
                elif phase == "descend":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.02 or phase_t >= phase_timeout: phase, phase_t = "grasp", 0
                elif phase == "grasp":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = 1.0
                    if phase_t >= grasp_dwell: phase, phase_t = "lift", 0
                elif phase == "lift":
                    target_raw = np.array([p_obj[0], p_obj[1], grasp_target_z + lift]); grip = 1.0
                    if abs(p_eef[2] - (grasp_target_z + lift)) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "transit", 0
                elif phase == "transit":
                    target_raw = np.array([basket_xy[0], basket_xy[1], grasp_target_z + lift]); grip = 1.0
                    if np.linalg.norm(target_raw[:2] - p_eef[:2]) < 0.04 or phase_t >= phase_timeout: phase, phase_t = "drop_descend", 0
                elif phase == "drop_descend":
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = 1.0
                    if abs(p_eef[2] - (basket_z + drop_offset)) < 0.035 or phase_t >= phase_timeout: phase, phase_t = "release", 0
                else:  # release
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = -1.0

                if blend:
                    if phase != prev_phase:
                        blend_from = target_raw.copy(); blend_step = 0
                    if blend_step < blend_window:
                        alpha = (blend_step + 1) / blend_window
                        target = blend_from + alpha * (target_raw - blend_from)
                        blend_step += 1
                    else:
                        target = target_raw
                else:
                    target = target_raw

                delta = np.clip(kp * (target - p_eef), -1.0, 1.0)
                if prev_action_xyz is not None:
                    jump = float(np.linalg.norm(delta - prev_action_xyz))
                    if jump > max_jump: max_jump, jump_phase = jump, f"{prev_phase}->{phase}" if prev_phase != phase else phase
                prev_action_xyz = delta
                n_sat += int(np.any(np.isclose(np.abs(delta), 1.0)))
                n_steps += 1
                if prev_phase != phase: phase_seq.append((t, prev_phase, phase))

                action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip], dtype=np.float32)
                obs, r, done, info = env.step(action.tolist())
                phase_t += 1; t += 1
                if phase == "release" and phase_t > release_dwell: break

            final_obj = obj_filter(obs).get(chosen, p_obj)
            final_xy_d = float(np.linalg.norm(np.asarray(final_obj[:2]) - basket_xy))
            final_z = float(final_obj[2])
            success = final_xy_d < success_xy_radius and final_z < success_z_max
            env.close()

            is_shove = bool(success and max_lift < grasp_lift_thresh)
            if success:
                kept += 1
                if is_shove: shoves += 1
            episodes.append({"task": ti, "ep": ep, "chosen": chosen, "success": success,
                             "final_xy_d": round(final_xy_d, 3), "max_lift": round(max_lift, 3),
                             "is_shove": is_shove, "frac_saturated": round(n_sat / max(n_steps, 1), 3),
                             "max_action_jump": round(max_jump, 3), "jump_at": jump_phase,
                             "n_steps": n_steps, "final_phase_reached": phase_seq[-1][2] if phase_seq else phase})
            print(f"[diag task {ti} ep {ep}] chosen={chosen} success={success} max_lift={max_lift:.3f} "
                 f"is_shove={is_shove} frac_saturated={n_sat/max(n_steps,1):.2f} "
                 f"max_action_jump={max_jump:.3f}@{jump_phase} n_steps={n_steps}")

    succ_eps = [e for e in episodes if e["success"]]
    summary = {
        "attempted": attempted, "kept": kept, "kept_frac": round(kept / max(attempted, 1), 3),
        "shoves_among_kept": shoves,
        "shove_frac_of_kept": round(shoves / max(kept, 1), 3),
        "mean_max_lift_all": round(float(np.mean([e["max_lift"] for e in episodes])), 3) if episodes else None,
        "mean_max_lift_kept": round(float(np.mean([e["max_lift"] for e in succ_eps])), 3) if succ_eps else None,
        "mean_frac_saturated_all": round(float(np.mean([e["frac_saturated"] for e in episodes])), 3) if episodes else None,
        "mean_frac_saturated_kept": round(float(np.mean([e["frac_saturated"] for e in succ_eps])), 3) if succ_eps else None,
        "mean_max_action_jump_all": round(float(np.mean([e["max_action_jump"] for e in episodes])), 3) if episodes else None,
        "mean_max_action_jump_kept": round(float(np.mean([e["max_action_jump"] for e in succ_eps])), 3) if succ_eps else None,
        "grasp_lift_thresh": grasp_lift_thresh,
        "verdict": ("SHOVE-CONTAMINATED: a large fraction of 'successful' kept demos never lifted "
                    "the object -- these teach the model that shoving passes for pick-place"
                    if shoves / max(kept, 1) > 0.15 else
                    "grasp verification looks OK; smoothness/saturation may still be the issue"),
    }
    json.dump({"summary": summary, "episodes": episodes},
             open(f"{VOL_PATH}/libero_decorrelated_demo_diagnostic.json", "w"), indent=2)
    vol.commit()
    print("DIAGNOSTIC RESULT:", json.dumps(summary, indent=2))
    return summary


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=1800)
def rotation_collapse_diagnostic(n_frames: int = 100000, res: int = 64, horizon: int = 8,
                                 n_eval_samples: int = 500,
                                 ckpts: str = "ckpt_linear_rat_vit_s0_v2.pt,"
                                              "ckpt_linear_rat_vit_s0_v2_decorr.pt,"
                                              "ckpt_linear_rat_vit_s0_v2_decorr_blend.pt,"
                                              "ckpt_decorr_s2000_rf85.pt"):
    """Tests the rotation-zeroing hypothesis (DEVLOG cont.65 fan-out): every decorrelated
    demo frame hardcodes rotation action dims (indices 3,4,5) to EXACTLY 0.0
    (`libero_gen_decorrelated_demos`: `action = np.array([delta[0], delta[1], delta[2],
    0.0, 0.0, 0.0, grip])`), fed into 40-60% of every fine-tune minibatch for up to 8000
    steps under a plain per-element MSE loss (`action_head='linear'` -> `F.mse_loss(actions,
    target_actions)`, ChiVLA.forward) that weights all 7 action dims equally.

    (1) Quantifies what raw 0.0 normalizes to given the REAL data's own a_mu/a_sd (computed
    exactly as train_vla_libero_finetune_decorrelated does), and confirms the decorrelated
    demo pkl's rotation dims really are exactly 0.0 with zero variance.
    (2) THE DIRECT TEST: loads each checkpoint (pre-finetune baseline + several post-decorr-
    finetune variants) and compares each one's PREDICTED rotation-action variance across a
    fixed, diverse sample of REAL held-out states/images -- if fine-tuning on the zeroed-
    rotation demos taught the policy to suppress/flatten its rotation output regardless of
    visual conditioning (regression-to-a-constant), the finetuned checkpoints' rotation
    output std-across-inputs should collapse toward ~0 relative to the baseline, while
    translation/gripper dims (never zeroed in the demos) should NOT collapse the same way.
    CPU-only, no GPU, no LIBERO sim -- inference-only forward passes on a small model."""
    _bootstrap()
    import json, pickle
    import numpy as np
    import torch
    from collections import defaultdict
    from xvla.models.vla import ChiVLA, VLAConfig
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files

    dev = "cpu"; H = horizon
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))

    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out

    real_samples = build_samples(frames)
    d_a = real_samples[0][3].shape[1]; state_dim = real_samples[0][2].shape[0]
    A = np.stack([s[3] for s in real_samples]); S = np.stack([s[2] for s in real_samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    rot_idx = [3, 4, 5]; transl_idx = [0, 1, 2]; grip_idx = 6

    z0 = (0.0 - a_mu) / a_sd
    result = {"a_mu": np.round(a_mu, 4).tolist(), "a_sd": np.round(a_sd, 4).tolist(),
              "raw_zero_normalized_z0_all_dims": np.round(z0, 4).tolist(),
              "rotation_dims_a_mu": [round(float(a_mu[i]), 4) for i in rot_idx],
              "rotation_dims_a_sd": [round(float(a_sd[i]), 4) for i in rot_idx],
              "rotation_dims_z0_normalized": [round(float(z0[i]), 4) for i in rot_idx],
              "note_z0": ("normalized value the demo's hardcoded raw-0.0 rotation target takes "
                          "under the REAL a_mu/a_sd used at fine-tune time; MSE loss pulls the "
                          "model's normalized-space rotation prediction toward exactly this "
                          "constant for every decorrelated-demo minibatch row.")}

    # confirm decorrelated demo rotation dims are exactly 0.0 / zero variance
    decorr_stats = {}
    for pkl_name in ["libero_decorrelated_demos.pkl", "libero_decorrelated_demos_blend.pkl"]:
        try:
            df = pickle.load(open(f"{VOL_PATH}/{pkl_name}", "rb"))
            da = np.stack([f[4] for f in df])
            decorr_stats[pkl_name] = {
                "n_frames": len(df),
                "rotation_dims_min_max_std": [
                    {"dim": i, "min": round(float(da[:, i].min()), 6), "max": round(float(da[:, i].max()), 6),
                     "std": round(float(da[:, i].std()), 6)} for i in rot_idx],
                "translation_dims_std": [round(float(da[:, i].std()), 4) for i in transl_idx]}
        except FileNotFoundError:
            decorr_stats[pkl_name] = "not found"
    result["decorr_demo_action_stats"] = decorr_stats

    # ---- direct test: predicted-rotation-variance across real diverse inputs, per checkpoint ----
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    rng = np.random.RandomState(0)
    eval_idx = rng.choice(len(real_samples), size=min(n_eval_samples, len(real_samples)), replace=False)
    eval_samples = [real_samples[i] for i in eval_idx]
    imgs = torch.tensor(np.stack([s[0] for s in eval_samples])).permute(0, 3, 1, 2).float().div(255)
    instr = torch.tensor([encode(tasks.get(s[1], "")) for s in eval_samples])
    states = torch.tensor((np.stack([s[2] for s in eval_samples]) - s_mu) / s_sd, dtype=torch.float32)
    # ground-truth normalized targets for the SAME eval samples, as a reference scale
    tgt_actions = torch.tensor((np.stack([s[3] for s in eval_samples]) - a_mu) / a_sd, dtype=torch.float32)
    tgt_mean_ah = tgt_actions.mean(dim=1)  # (N, d_a): average over horizon per sample

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm="rational", qk_norm="rational", vision_encoder="vit")

    def eval_ckpt(ckpt_name):
        model = ChiVLA(cfg).to(dev)
        sd = torch.load(f"{VOL_PATH}/{ckpt_name}", map_location=dev, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        outs = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(eval_samples), bs):
                a, _ = model(imgs[i:i+bs], instr[i:i+bs], states[i:i+bs],
                             torch.zeros(min(bs, len(eval_samples) - i), dtype=torch.long))
                outs.append(a)
        A_pred = torch.cat(outs, dim=0)          # (N, H, d_a) normalized-space predictions
        A_pred_mean_h = A_pred.mean(dim=1)       # (N, d_a): per-sample mean over horizon

        def dim_report(idx_list, label):
            out = {}
            for i in idx_list:
                col = A_pred_mean_h[:, i]
                tgt = tgt_mean_ah[:, i]
                resid = col - tgt
                # Pearson correlation: does the prediction actually TRACK the correct
                # per-sample target (not just match its marginal variance)? A collapsed/
                # constant predictor can have near-target std by coincidence but ~0 correlation.
                cc = float(col.std()) * float(tgt.std())
                corr = float(((col - col.mean()) * (tgt - tgt.mean())).mean() / (cc + 1e-8))
                out[f"dim{i}"] = {
                    "pred_std_across_inputs": round(float(col.std()), 5),
                    "pred_mean": round(float(col.mean()), 5),
                    "target_std_across_inputs": round(float(tgt.std()), 5),
                    "pred_over_target_std_ratio": round(float(col.std() / (tgt.std() + 1e-8)), 5),
                    "rmse_normalized": round(float((resid ** 2).mean().sqrt()), 5),
                    "pearson_corr_pred_vs_target": round(corr, 5),
                }
            return out

        return {
            "rotation": dim_report(rot_idx, "rotation"),
            "translation": dim_report(transl_idx, "translation"),
            "gripper": dim_report([grip_idx], "gripper"),
        }

    ckpt_list = [c.strip() for c in ckpts.split(",") if c.strip()]
    per_ckpt = {}
    for ck in ckpt_list:
        try:
            print(f"evaluating {ck} ...")
            per_ckpt[ck] = eval_ckpt(ck)
            print(f"  {ck}: rotation={per_ckpt[ck]['rotation']}")
        except Exception as e:
            per_ckpt[ck] = {"error": str(e)}
            print(f"  {ck}: ERROR {e}")
    result["n_eval_samples"] = len(eval_samples)
    result["per_checkpoint"] = per_ckpt

    with open(f"{VOL_PATH}/rotation_collapse_diagnostic.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("ROTATION_COLLAPSE_RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def train_vla_libero_finetune_decorrelated(ckpt_in: str = "ckpt_linear_rat_vit_s0_v2.pt",
                                           ckpt_out: str = "ckpt_linear_rat_vit_s0_v2_decorr.pt",
                                           steps: int = 8000, lr: float = 2e-4, real_frac: float = 0.6,
                                           n_frames: int = 100000, res: int = 64, horizon: int = 8,
                                           decorr_pkl: str = "libero_decorrelated_demos.pkl",
                                           decorr_tasks: str = "libero_decorrelated_tasks.json",
                                           norm: str = "rational", vision_encoder: str = "vit",
                                           batch: int = 256):
    """The GPU half of the instruction-swap fix (DEVLOG cont.59/60): fine-tune the already-
    trained 94.5%/94.8%-mean checkpoint on a MIX of the real teleop data and the decorrelated
    scripted demos (libero_gen_decorrelated_demos), so scene-identity is no longer a perfect
    predictor of instruction. real_frac of each minibatch comes from the real cache, the rest
    from the augmented set -- RLPD-style mixed sampling (Ball et al., arXiv:2311.05067), matching
    NEXT_PHASE_PLAN.md section 2.5 step 5's proposed pattern, here applied to BC fine-tuning
    rather than RL. Both instruction sets share an IDENTICAL vocabulary (verified: the augmented
    instructions introduce zero new words -- every distractor name is some other task template's
    real target, the same closed 10-object set) so `tok_emb`'s embedding table needs no resizing
    and `ckpt_in`'s weights load with zero shape mismatches. Normalization statistics (a_mu/a_sd/
    s_mu/s_sd) are computed from the REAL data only and reused for the augmented data, since the
    loaded checkpoint's action_head/state_proj already expect that scale."""
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

    decorr_tasks_d = json.load(open(f"{VOL_PATH}/{decorr_tasks}"))
    decorr_tasks_d = {int(k): v for k, v in decorr_tasks_d.items()}
    tasks_all = {**tasks, **decorr_tasks_d}
    words = set()
    for t in tasks_all.values(): words.update(t.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words): vocab[w] = len(vocab)
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    real_vocab_words = set()
    for t in tasks.values(): real_vocab_words.update(t.lower().replace(".", "").split())
    new_words = words - real_vocab_words
    assert not new_words, f"decorrelated instructions introduced new vocab words: {new_words}"

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out

    real_samples = build_samples(frames)
    d_a = real_samples[0][3].shape[1]; state_dim = real_samples[0][2].shape[0]
    A = np.stack([s[3] for s in real_samples]); S = np.stack([s[2] for s in real_samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"real: {len(real_samples)} samples, d_a={d_a} state_dim={state_dim}")

    decorr_frames = pickle.load(open(f"{VOL_PATH}/{decorr_pkl}", "rb"))
    decorr_samples_raw = build_samples(decorr_frames)
    # decorrelated demos' state may have a different raw width than the real dataset's own
    # observation.state field (both come from the same build_state-style eef/quat/gripper
    # convention, but pad/truncate defensively rather than assume exact equality).
    def fix_state(s):
        return s[:state_dim] if len(s) >= state_dim else np.pad(s, (0, state_dim - len(s)))
    decorr_samples = [(im, ti, fix_state(st), ac[:, :d_a]) for (im, ti, st, ac) in decorr_samples_raw]
    print(f"decorrelated: {len(decorr_samples)} samples, {len(set(s[1] for s in decorr_samples))} distinct (task,obj) pairs")

    def to_tensors(samples):
        imgs = torch.tensor(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
        instr = torch.tensor([encode(tasks_all.get(s[1], "")) for s in samples], device=dev)
        states = torch.tensor((np.stack([s[2] for s in samples]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
        actions = torch.tensor((np.stack([s[3] for s in samples]) - a_mu) / a_sd, dtype=torch.float32, device=dev)
        return imgs, instr, states, actions

    real_imgs, real_instr, real_states, real_actions = to_tensors(real_samples)
    dec_imgs, dec_instr, dec_states, dec_actions = to_tensors(decorr_samples)

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_in}", map_location=dev, weights_only=True))
    print(f"loaded {ckpt_in}, continuing training {steps} steps, real_frac={real_frac}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=steps, warmup_frac=0.05)
    n_real = max(1, round(batch * real_frac)); n_dec = batch - n_real
    model.train()
    for step in range(steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
        ridx = torch.randint(len(real_samples), (n_real,), device=dev)
        didx = torch.randint(len(decorr_samples), (n_dec,), device=dev)
        imgs_b = torch.cat([real_imgs[ridx], dec_imgs[didx]])
        instr_b = torch.cat([real_instr[ridx], dec_instr[didx]])
        states_b = torch.cat([real_states[ridx], dec_states[didx]])
        actions_b = torch.cat([real_actions[ridx], dec_actions[didx]])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs_b, instr_b, states_b,
                           torch.zeros(batch, dtype=torch.long, device=dev),
                           target_actions=actions_b, progress=step / max(steps, 1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0: print(f"  step {step} loss {loss.item():.4f}")
    model.eval()
    torch.save(model.state_dict(), f"{VOL_PATH}/{ckpt_out}")
    vol.commit()
    result = {"ckpt_in": ckpt_in, "ckpt_out": ckpt_out, "steps": steps, "real_frac": real_frac,
              "n_real_samples": len(real_samples), "n_decorr_samples": len(decorr_samples),
              "final_loss": round(float(loss.item()), 4)}
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def dagger_gen_corrective(ckpt_in: str = "ckpt_linear_rat_conv_s0.pt", vision_encoder: str = "conv",
                          norm: str = "rational", pilot_tasks: str = "3,5", eps_per_task_pre: int = 20,
                          max_corrections_per_task: int = 15, max_steps_eval: int = 280,
                          num_steps_wait: int = 10, res: int = 64, horizon: int = 8,
                          n_frames: int = 100000, kp: float = 8.0, hover: float = 0.10,
                          grasp_z_offset: float = 0.0, grasp_dwell: int = 15, lift: float = 0.15,
                          drop_offset: float = 0.08, release_dwell: int = 10, phase_timeout: int = 60,
                          blend_window: int = 18, max_jump_accept: float = 0.6,
                          grasp_lift_thresh: float = 0.05, tag: str = "_dagger_pilot"):
    """STAGE 1 of the (now 2-stage) online_dagger pilot. Split out of a single combined function
    after that version reproducibly segfaulted (SIGSEGV) THREE TIMES IN A ROW at the exact same
    location -- bulk GPU tensor conversion of a batch of images (torch.tensor(np.stack(...)).to
    (dev)), done in a process where a LIBERO/MuJoCo env had already been created earlier in that
    same process. Two different in-process fixes (reordering the tensor-vs-env sequence; reducing
    env creation count via per-task env reuse) did NOT change the outcome -- the crash point was
    identical each time. The one thing every OTHER working function in this codebase already does
    right, and this one didn't: bulk image-tensor GPU conversion (train_vla_libero_finetune_
    decorrelated) and live LIBERO rollouts (libero_gen_decorrelated_demos) never happen in the SAME
    process. Splitting into two Modal functions (this one: envs + only small per-step tensor ops,
    proven safe; the other, dagger_finetune_eval: bulk tensor ops strictly before any env is
    created) is the fix that actually matches the evidence, rather than another guess.

    This stage: (1) pre-eval `pilot_tasks` at eps_per_task_pre using canonical init states with the
    CURRENT checkpoint, record which (task, episode) fail; (2) for up to max_corrections_per_task
    failing episodes per task, replay the SAME seed/init-state with the blend-fixed (BLEND_WINDOW,
    cont.65) scripted P-controller aimed at the task's TRUE bddl target object, quality-gate the
    resulting demo (reject on excess max_action_jump or the shove-not-grasp signature), keep it if
    it passes. Saves {pre_eval, fail_eps, corr_frames, corr_diag, a_mu/a_sd/s_mu/s_sd, d_a,
    state_dim} to a pickle on the volume for dagger_finetune_eval to consume -- no bulk image
    tensor is ever built in this process.

    HONEST FLAG, carried from the original single-function docstring: a full-dosage version of
    this general idea (scripted-demo retraining) already collapsed capability once (94.5%->6.5%)
    even after this identical blend fix, and the rotation-zeroing hypothesis for that collapse was
    empirically REFUTED, not confirmed -- the root cause is still not fully understood. This
    pilot's mitigations (far lower dosage, quality gating, explicit held-out split in stage 2) are
    real but NOT proven sufficient -- a regression in stage 2 should be read as a stop signal."""
    _bootstrap()
    import json, os, re, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image

    dev = "cuda"; H = horizon
    pilot_task_idxs = [int(x) for x in str(pilot_tasks).split(",")]
    name = "lerobot/libero_object_image"

    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    def obj_filter(obs):
        return {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
               if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
               and "_to_" not in k and "eef" not in k}

    def build_state_full(obs):
        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)

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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out

    real_samples = build_samples(frames)
    d_a = real_samples[0][3].shape[1]; state_dim = real_samples[0][2].shape[0]
    A = np.stack([s[3] for s in real_samples]); S = np.stack([s[2] for s in real_samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    print(f"real: {len(real_samples)} samples (stats only, no bulk image tensor built), d_a={d_a} state_dim={state_dim}")
    del real_samples, frames  # free -- this process never needs the bulk real image data

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_in}", map_location=dev, weights_only=True))
    model.eval()
    print(f"loaded {ckpt_in} (vision_encoder={vision_encoder} norm={norm})")

    def build_state(obs):
        v = build_state_full(obs)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    @torch.no_grad()
    def rollout_episode(env, task, init_states, ep):
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env.seed(ep); obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
        ok = False; t = 0
        while t < max_steps_eval and not ok:
            im, st = obs_tensors(obs)
            a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
            chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
            for k in range(min(H, H)):
                act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                obs, r, done, info = env.step(act.tolist()); t += 1
                if r > 0: ok = True
                if done or ok or t >= max_steps_eval: break
        return ok

    pre_eval_per_task = {}; fail_eps = {}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        succ = 0; fails = []
        for ep in range(eps_per_task_pre):
            ok = rollout_episode(env, task, init_states, ep)
            succ += int(ok)
            if not ok: fails.append(ep)
        env.close()
        pre_eval_per_task[ti] = {"language": task.language, "success": round(succ / eps_per_task_pre, 3), "n": eps_per_task_pre}
        fail_eps[ti] = fails
        print(f"[pre-dagger task {ti}] {task.language}: {succ}/{eps_per_task_pre}")
    pre_eval = {"per_task": pre_eval_per_task,
               "overall": round(float(np.mean([v["success"] for v in pre_eval_per_task.values()])), 3)}
    print("PRE-DAGGER EVAL:", json.dumps(pre_eval, indent=2))

    corr_frames = []
    corr_diag = {"attempted": 0, "kept": 0, "rejected_jump": 0, "rejected_shove": 0, "rejected_no_obj": 0}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        eps_to_fix = fail_eps.get(ti, [])[:max_corrections_per_task]
        if not eps_to_fix:
            continue
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        for ep in eps_to_fix:
            corr_diag["attempted"] += 1
            env.seed(ep); obs = env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)

            obj_pos0 = obj_filter(obs)
            basket_key = next((k for k in obs if "basket" in k and k.endswith("_pos")), None)
            chosen = next((k for k in obj_pos0 if readable(k) in task.language.lower()), None)
            if chosen is None or basket_key is None:
                corr_diag["rejected_no_obj"] += 1; continue
            obj_z0 = float(obj_pos0[chosen][2]); grasp_target_z = obj_z0
            basket_xy = np.asarray(obs[basket_key][:2]); basket_z = float(obs[basket_key][2])

            ep_frames = []; max_lift = -1e9; max_jump = 0.0; prev_delta = None
            phase = "hover"; phase_t = 0; t = 0
            blend_from = None; blend_step = blend_window
            episode_index = 2_000_000 + ti * 10000 + ep
            task_index = 2000 + ti
            while t < 250:
                obj_pos_t = obj_filter(obs)
                p_obj = np.asarray(obj_pos_t.get(chosen, obj_pos0[chosen]))
                p_eef = np.asarray(obs["robot0_eef_pos"])
                max_lift = max(max_lift, float(p_obj[2]) - obj_z0)

                phase_before = phase
                if phase == "hover":
                    target_raw = p_obj + np.array([0.0, 0.0, hover]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "descend", 0
                elif phase == "descend":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.02 or phase_t >= phase_timeout: phase, phase_t = "grasp", 0
                elif phase == "grasp":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = 1.0
                    if phase_t >= grasp_dwell: phase, phase_t = "lift", 0
                elif phase == "lift":
                    target_raw = np.array([p_obj[0], p_obj[1], grasp_target_z + lift]); grip = 1.0
                    if abs(p_eef[2] - (grasp_target_z + lift)) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "transit", 0
                elif phase == "transit":
                    target_raw = np.array([basket_xy[0], basket_xy[1], grasp_target_z + lift]); grip = 1.0
                    if np.linalg.norm(target_raw[:2] - p_eef[:2]) < 0.04 or phase_t >= phase_timeout: phase, phase_t = "drop_descend", 0
                elif phase == "drop_descend":
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = 1.0
                    if abs(p_eef[2] - (basket_z + drop_offset)) < 0.035 or phase_t >= phase_timeout: phase, phase_t = "release", 0
                else:
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = -1.0

                if phase != phase_before:
                    blend_from = target_raw.copy(); blend_step = 0
                if blend_step < blend_window:
                    alpha = (blend_step + 1) / blend_window
                    target = blend_from + alpha * (target_raw - blend_from)
                    blend_step += 1
                else:
                    target = target_raw

                delta = np.clip(kp * (target - p_eef), -1.0, 1.0)
                if prev_delta is not None:
                    max_jump = max(max_jump, float(np.linalg.norm(delta - prev_delta)))
                prev_delta = delta

                action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip], dtype=np.float32)
                img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)), dtype=np.uint8)
                state = build_state_full(obs)
                ep_frames.append((episode_index, t, img, state, action, task_index))
                obs, r, done, info = env.step(action.tolist())
                phase_t += 1; t += 1
                if phase == "release" and phase_t > release_dwell: break

            final_obj = obj_filter(obs).get(chosen, p_obj)
            final_xy_d = float(np.linalg.norm(np.asarray(final_obj[:2]) - basket_xy))
            final_z = float(final_obj[2])
            geo_success = final_xy_d < 0.12 and final_z < 0.15
            is_shove = geo_success and max_lift < grasp_lift_thresh

            if not geo_success:
                corr_diag["rejected_shove"] += 1 if is_shove else 0
                continue
            if is_shove:
                corr_diag["rejected_shove"] += 1; continue
            if max_jump > max_jump_accept:
                corr_diag["rejected_jump"] += 1; continue
            corr_frames.extend(ep_frames)
            corr_diag["kept"] += 1
        env.close()
    print("CORRECTIVE DEMO GENERATION:", json.dumps(corr_diag, indent=2))

    out_pkl = f"dagger_corrective{tag}.pkl"
    payload = {"pre_eval": pre_eval, "fail_eps": fail_eps, "corr_frames": corr_frames,
              "corr_diag": corr_diag, "pilot_task_idxs": pilot_task_idxs,
              "a_mu": a_mu, "a_sd": a_sd, "s_mu": s_mu, "s_sd": s_sd,
              "d_a": d_a, "state_dim": state_dim,
              "task_languages": {ti: suite.get_task(ti).language for ti in pilot_task_idxs}}
    pickle.dump(payload, open(f"{VOL_PATH}/{out_pkl}", "wb"))
    vol.commit()
    print(f"saved -> {out_pkl}  (n_corr_frames={len(corr_frames)}, n_kept_episodes={corr_diag['kept']})")
    return {"pre_eval": pre_eval, "corrective_demo_stats": corr_diag, "out_pkl": out_pkl}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=6 * 3600)
def dagger_finetune_eval(ckpt_in: str = "ckpt_linear_rat_conv_s0.pt", vision_encoder: str = "conv",
                         norm: str = "rational", eps_per_task_pre: int = 20,
                         eps_per_task_post: int = 50, max_steps_eval: int = 280,
                         num_steps_wait: int = 10, res: int = 64, horizon: int = 8,
                         n_frames: int = 100000, mix_frac: float = 0.12, finetune_steps: int = 2000,
                         lr: float = 5e-5, batch: int = 256, ckpt_out: str = "ckpt_dagger_pilot.pt",
                         corrective_pkl: str = "dagger_corrective_dagger_pilot.pkl",
                         tag: str = "_dagger_pilot", pre_eval_seed_key: str = ""):
    """STAGE 2 (see dagger_gen_corrective's docstring for why this is split out). Builds ALL bulk
    GPU image tensors (real cache + corrective demos, loaded from `corrective_pkl`) strictly BEFORE
    any LIBERO env is created in this process, fine-tunes `ckpt_in` on a low-mix-fraction blend of
    real + corrective data, THEN runs post-eval (env creation only now, after all bulk tensor
    work -- the ordering proven safe by every successful pre-eval run in stage 1 and by this same
    function's own real-tensor-then-model-load-then-envs sequence).

    A further fix after this exact function ALSO segfaulted once (SIGSEGV right after model
    loading, before the first training step -- i.e. with ZERO LIBERO envs yet created in that
    process, ruling out the earlier per-episode/per-env-count theories entirely): `libero`/
    `robosuite` are NOT imported at all until after the fine-tune loop completes, right before
    post-eval actually needs them. These packages' EGL/OpenGL bindings appear to do some global
    initialization at IMPORT time (not just at env-creation time) that can corrupt CUDA/PyTorch
    memory state later in the same process -- import-time side effects, not instance creation
    count, are the more precise culprit. Every prior crash in this pilot's development is
    consistent with this: the only thing that changed across all 4 failures was WHEN in the
    process robosuite/libero had already been imported relative to heavy CUDA tensor work, not
    how many envs were created or in what order relative to tensor building alone."""
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

    payload = pickle.load(open(f"{VOL_PATH}/{corrective_pkl}", "rb"))
    if "pre_eval" not in payload:
        # Compatibility shim for ensemble_dagger_gen_corrective's payload schema (per_episode_bits
        # keyed by seed, not a single pre_eval dict) -- derive THIS seed's own pre_eval/fail_eps
        # from its own per-episode bits, not the ensemble's, so post_eval is compared against the
        # correct baseline for whichever checkpoint is actually being continued here. Done INSIDE
        # this container (not by locally patching the pickle) to avoid a real numpy pickle-version
        # mismatch found this session: locally re-pickling with a newer numpy than this image's
        # produced "ModuleNotFoundError: No module named 'numpy._core.numeric'" on load.
        assert pre_eval_seed_key, "payload has no 'pre_eval' and no pre_eval_seed_key given -- cannot derive a baseline"
        bits_by_task = payload["per_episode_bits"]
        per_task = {}
        for ti, bits in bits_by_task.items():
            seed_bits = bits[pre_eval_seed_key]
            n = len(seed_bits)
            per_task[ti] = {"language": payload["task_languages"][ti],
                           "success": round(sum(seed_bits) / n, 3), "n": n}
        overall = round(float(sum(v["success"] for v in per_task.values()) / len(per_task)), 3)
        payload["pre_eval"] = {"per_task": per_task, "overall": overall}
        payload["fail_eps"] = {ti: [j for j, ok in enumerate(bits[pre_eval_seed_key]) if not ok]
                               for ti, bits in bits_by_task.items()}
        print(f"derived pre_eval for seed_key={pre_eval_seed_key} from per_episode_bits: {payload['pre_eval']}")
    pre_eval = payload["pre_eval"]; corr_frames = payload["corr_frames"]
    corr_diag = payload["corr_diag"]; pilot_task_idxs = payload["pilot_task_idxs"]
    a_mu, a_sd, s_mu, s_sd = payload["a_mu"], payload["a_sd"], payload["s_mu"], payload["s_sd"]
    d_a, state_dim = payload["d_a"], payload["state_dim"]
    print(f"loaded {corrective_pkl}: pre_eval overall={pre_eval['overall']}, "
         f"corrective kept episodes={corr_diag['kept']}, n_corr_frames={len(corr_frames)}")

    if corr_diag["kept"] == 0:
        result = {"pre_eval": pre_eval, "corrective_demo_stats": corr_diag,
                 "note": "no corrective demos survived quality gating -- aborting fine-tune, nothing to compare."}
        json.dump(result, open(f"{VOL_PATH}/dagger_pilot{tag}.json", "w"), indent=2)
        vol.commit()
        return result

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
    tasks_all = {**tasks, **{2000 + ti: lang for ti, lang in payload["task_languages"].items()}}
    words = set()
    for t in tasks_all.values(): words.update(t.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for w in sorted(words): vocab[w] = len(vocab)
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out
    real_samples = build_samples(frames)

    corr_eps = defaultdict(list)
    for f in corr_frames: corr_eps[f[0]].append(f)
    corr_samples_raw = []
    for ep, fs in corr_eps.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            corr_samples_raw.append((fs[i][0], fs[i][1], fs[i][2], fs[i][3],
                                     np.stack([fs[i + k][4] for k in range(H)]), fs[i][5]))
    print(f"corrective: {len(corr_samples_raw)} H-step samples from {corr_diag['kept']} kept episodes")

    # ---- ALL bulk GPU image-tensor conversion happens HERE, before any LIBERO env exists in
    # this process (see dagger_gen_corrective's docstring for why this ordering is load-bearing). ----
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    real_imgs = torch.tensor(np.stack([s[0] for s in real_samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    real_instr = torch.tensor([encode(tasks.get(s[1], "")) for s in real_samples], device=dev)
    real_states = torch.tensor((np.stack([s[2] for s in real_samples]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    real_actions = torch.tensor((np.stack([s[3] for s in real_samples]) - a_mu) / a_sd, dtype=torch.float32, device=dev)
    corr_imgs = torch.tensor(np.stack([s[2] for s in corr_samples_raw])).permute(0, 3, 1, 2).float().div(255).to(dev)
    corr_instr = torch.tensor([encode(tasks_all.get(s[5], "")) for s in corr_samples_raw], device=dev)
    corr_states = torch.tensor((np.stack([s[3] for s in corr_samples_raw]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
    corr_actions = torch.tensor((np.stack([s[4] for s in corr_samples_raw]) - a_mu) / a_sd, dtype=torch.float32, device=dev)
    del real_samples, frames, corr_frames, corr_eps

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    model = ChiVLA(cfg).to(dev)
    model.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_in}", map_location=dev, weights_only=True))
    print(f"loaded {ckpt_in} (vision_encoder={vision_encoder} norm={norm}), continuing training")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=finetune_steps, warmup_frac=0.05)
    n_corr = max(1, round(batch * mix_frac)); n_real = batch - n_corr
    model.train()
    for step in range(finetune_steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
        ridx = torch.randint(len(real_imgs), (n_real,), device=dev)
        cidx = torch.randint(len(corr_imgs), (n_corr,), device=dev)
        imgs_b = torch.cat([real_imgs[ridx], corr_imgs[cidx]])
        instr_b = torch.cat([real_instr[ridx], corr_instr[cidx]])
        states_b = torch.cat([real_states[ridx], corr_states[cidx]])
        actions_b = torch.cat([real_actions[ridx], corr_actions[cidx]])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(imgs_b, instr_b, states_b,
                           torch.zeros(batch, dtype=torch.long, device=dev),
                           target_actions=actions_b, progress=step / max(finetune_steps, 1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0: print(f"  [dagger finetune] step {step} loss {loss.item():.4f}")
    model.eval()
    torch.save(model.state_dict(), f"{VOL_PATH}/{ckpt_out}")
    del real_imgs, real_instr, real_states, real_actions, corr_imgs, corr_instr, corr_states, corr_actions
    torch.cuda.empty_cache()

    # ---- post-eval: libero/robosuite are imported HERE for the first time in this process --
    # not at the top of the function -- and env creation happens only now, after all bulk tensor
    # work is done. See the docstring for why the import location itself, not just env creation
    # count, is the fix. ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    @torch.no_grad()
    def rollout_episode(env, task, init_states, ep):
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env.seed(ep); obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
        ok = False; t = 0
        while t < max_steps_eval and not ok:
            im, st = obs_tensors(obs)
            a, _ = model(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
            chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
            for k in range(min(H, H)):
                act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                obs, r, done, info = env.step(act.tolist()); t += 1
                if r > 0: ok = True
                if done or ok or t >= max_steps_eval: break
        return ok

    post_eval_per_task = {}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        succ = 0
        for ep in range(eps_per_task_post):
            succ += int(rollout_episode(env, task, init_states, ep))
        env.close()
        post_eval_per_task[ti] = {"language": task.language, "success": round(succ / eps_per_task_post, 3), "n": eps_per_task_post}
        print(f"[post-dagger task {ti}] {task.language}: {succ}/{eps_per_task_post}")
    post_eval = {"per_task": post_eval_per_task,
                "overall": round(float(np.mean([v["success"] for v in post_eval_per_task.values()])), 3)}
    print("POST-DAGGER EVAL:", json.dumps(post_eval, indent=2))

    overlap_n = eps_per_task_pre
    held_out_note = (f"post_eval uses eps_per_task_post={eps_per_task_post} episodes per task; "
                     f"episodes 0..{overlap_n - 1} sit on the same init states corrective demos "
                     f"were sourced from (expected to improve -- direct fix, not evidence of "
                     f"generalization); episodes {overlap_n}..{eps_per_task_post - 1} are the "
                     f"genuinely held-out generalization check." if eps_per_task_post > overlap_n
                     else "eps_per_task_post <= eps_per_task_pre: no held-out slice, full overlap.")

    result = {"config": {"ckpt_in": ckpt_in, "vision_encoder": vision_encoder, "norm": norm,
                        "pilot_tasks": pilot_task_idxs, "eps_per_task_pre": eps_per_task_pre,
                        "eps_per_task_post": eps_per_task_post, "mix_frac": mix_frac,
                        "finetune_steps": finetune_steps, "lr": lr},
             "pre_eval": pre_eval, "post_eval": post_eval, "corrective_demo_stats": corr_diag,
             "held_out_caveat": held_out_note,
             "go_no_go": ("GO" if post_eval["overall"] >= pre_eval["overall"] else "NO-GO: regression vs pre-eval"),
             "honest_note": ("Full-episode replay from the failing episode's own init state, not "
                             "exact mid-trajectory sim-state forking. A prior full-dosage version "
                             "of this general idea collapsed capability once even after this same "
                             "blend fix; this pilot's mitigations (low dosage, low mix_frac, quality "
                             "gating, held-out split) are real but unproven -- a regression here "
                             "should be read as a stop signal for this direction, not retuned around.")}
    json.dump(result, open(f"{VOL_PATH}/dagger_pilot{tag}.json", "w"), indent=2)
    vol.commit()
    print("RESULT:", json.dumps({"pre_eval": pre_eval, "post_eval": post_eval,
                                 "go_no_go": result["go_no_go"]}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=5 * 3600)
def ensemble_consensus_pilot(ckpts: str = "ckpt_linear_rat_conv_s0.pt,ckpt_linear_rat_conv_s1_matched.pt,ckpt_linear_rat_conv_s2_matched.pt",
                             vision_encoder: str = "conv", norm: str = "rational",
                             pilot_tasks: str = "0,3,5", eps_per_task: int = 20,
                             max_steps: int = 280, num_steps_wait: int = 10, res: int = 64,
                             horizon: int = 8, exec_h: int = 8, n_frames: int = 100000,
                             tag: str = ""):
    """ADAPTED first experiment for the inference_time_scaling lens from the sota-push-planning
    workflow. The original proposal was cross-seed consensus over the ProductRoutingHead's 2^G
    mode zonotope (enumerate_modes) -- that needs product-head checkpoints trained to
    matched-protocol quality, which do NOT currently exist (the only product-head checkpoints on
    the volume are the ~72-89% BC-pretrain-from-scratch ones inside the AWR runs, not independent
    matched-protocol seeds). Rather than train 3 new product-head seeds just to enable literal
    zonotope consensus (no longer 'zero GPU-training hours' if we did), this runs the nearest
    zero-training analog with checkpoints that actually exist: ELEMENTWISE PREDICTION AVERAGING
    across the 3 already-trained matched-protocol conv+rational seeds (93.7%+-0.8pp mean), a
    standard, well-founded ensemble method (model averaging / 'wisdom of committees'), still zero
    additional GPU-training, still bounded downside (worst case it's close to the single-seed
    mean, not worse than any individual model by construction of averaging non-adversarial
    predictions). This is a genuine adaptation of the original idea, not the literal zonotope
    consensus method -- flagged explicitly, not silently substituted.

    Protocol: on a reduced slice (pilot_tasks x eps_per_task, default 3 tasks x 20 trials = 60
    episodes, matching the lens's own 'screening slice' recommendation), run each of the 3
    checkpoints INDIVIDUALLY (paired, same canonical init states) to get a same-slice single-seed
    baseline, then run the ENSEMBLE policy (average the 3 models' predicted action chunks
    elementwise in raw action units at every replan step, execute exec_h steps, replan) on the
    IDENTICAL episodes. Compares ensemble success vs. the mean of the 3 paired single-seed runs on
    the same states -- a fair paired comparison, not a comparison against the previously-reported
    full 500-trial matched-protocol numbers (different episodes/seeds)."""
    _bootstrap()
    import json, os
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image

    dev = "cuda"; H = horizon
    ckpt_list = [c.strip() for c in str(ckpts).split(",")]
    pilot_task_idxs = [int(x) for x in str(pilot_tasks).split(",")]
    name = "lerobot/libero_object_image"

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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    import pickle
    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    models = []
    for c in ckpt_list:
        m = ChiVLA(cfg).to(dev)
        m.load_state_dict(torch.load(f"{VOL_PATH}/{c}", map_location=dev, weights_only=True))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} checkpoints: {ckpt_list}")

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    @torch.no_grad()
    def rollout(env, task, init_states, ep, mode, model_idx=None):
        # reuses a single, already-open env across ALL episodes/conditions of a task -- creating
        # a fresh OffScreenRenderEnv per episode (the original version of this function, and of
        # dagger_pilot before its own fix) is a confirmed source of a MuJoCo/EGL resource leak
        # that manifests as either an outright segfault or, worse, SILENT intermittent render/
        # state corruption -- plausibly the explanation for an anomalous first run of this
        # function where 2 of 3 seeds cratered on task 0 (6/20) versus their known-good
        # matched-protocol numbers (96%/88%), while the 3rd stayed near-expected. That run's
        # numbers were discarded, not trusted, once this was identified.
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env.seed(ep); obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
        ok = False; t = 0
        while t < max_steps and not ok:
            im, st = obs_tensors(obs)
            if mode == "single":
                a, _ = models[model_idx](im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
            else:  # ensemble: elementwise average of all models' predicted chunks (raw action units)
                chunks = []
                for m in models:
                    a, _ = m(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                    chunks.append((a[0] * a_sd_t + a_mu_t).cpu().numpy())
                chunk = np.mean(chunks, axis=0)
            for k in range(min(exec_h, H)):
                act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                obs, r, done, info = env.step(act.tolist()); t += 1
                if r > 0: ok = True
                if done or ok or t >= max_steps: break
        return ok

    single_results = {i: {} for i in range(len(models))}
    ensemble_results = {}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        for i in range(len(models)):
            succ = sum(int(rollout(env, task, init_states, ep, "single", i)) for ep in range(eps_per_task))
            single_results[i][ti] = round(succ / eps_per_task, 3)
            print(f"[single seed {i} task {ti}] {task.language}: {succ}/{eps_per_task}")
        succ = sum(int(rollout(env, task, init_states, ep, "ensemble")) for ep in range(eps_per_task))
        ensemble_results[ti] = round(succ / eps_per_task, 3)
        print(f"[ensemble task {ti}] {task.language}: {succ}/{eps_per_task}")
        env.close()

    single_seed_means = [round(float(np.mean(list(single_results[i].values()))), 4) for i in range(len(models))]
    mean_of_single_seed_means = round(float(np.mean(single_seed_means)), 4)
    ensemble_overall = round(float(np.mean(list(ensemble_results.values()))), 4)

    result = {"config": {"ckpts": ckpt_list, "pilot_tasks": pilot_task_idxs, "eps_per_task": eps_per_task,
                        "vision_encoder": vision_encoder, "norm": norm},
             "single_seed_per_task": single_results, "ensemble_per_task": ensemble_results,
             "single_seed_means": single_seed_means, "mean_of_single_seed_means": mean_of_single_seed_means,
             "ensemble_overall": ensemble_overall,
             "delta_vs_single_seed_mean": round(ensemble_overall - mean_of_single_seed_means, 4),
             "go_no_go": ("GO -- scale to full matched protocol" if ensemble_overall - mean_of_single_seed_means >= 0.02
                         else "NO-GO -- ensemble did not beat single-seed mean by >=2pp on this screening slice"),
             "adaptation_note": ("Elementwise prediction averaging across 3 already-trained matched-protocol "
                                 "conv+rational seeds, NOT the originally-proposed product-head mode-zonotope "
                                 "consensus (that needs matched-protocol-quality product-head checkpoints that "
                                 "don't exist yet) -- a genuine adaptation of the inference_time_scaling lens's "
                                 "idea to checkpoints that actually exist, flagged explicitly.")}
    out_name = f"ensemble_consensus_pilot{('_' + tag) if tag else ''}.json"
    json.dump(result, open(f"{VOL_PATH}/{out_name}", "w"), indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items()
                                 if k not in ("single_seed_per_task", "ensemble_per_task")}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=1800)
def weight_soup_interpolation_gate(ckpt_a: str = "ckpt_linear_rat_conv_s0.pt",
                                   ckpt_b: str = "ckpt_linear_rat_conv_s1_matched.pt",
                                   vision_encoder: str = "conv", norm: str = "rational",
                                   n_frames: int = 100000, res: int = 64, horizon: int = 8,
                                   recal_batch: int = 1024, eval_batch: int = 2048,
                                   alphas: str = "0,0.25,0.5,0.75,1", barrier_ratio: float = 1.5):
    """FREE (no LIBERO env, no rollout, just forward passes) pre-registered gate for the
    ensemble_distillation lens's top-ranked recommendation from the post-win-scaleup-planning
    workflow: before committing to a full closed-loop weight-souping experiment on the 3
    conv+rational seeds that just proved out as a +6.1pp ensemble (ensemble_consensus_pilot), check
    whether the two checkpoints are even linearly mode-connected in weight space. `libero_rollout_head`
    constructs a FRESH ChiVLA(cfg) with torch.manual_seed(seed) for every seed (confirmed by reading
    that function) -- seeds differ in BOTH init and minibatch order from step 0, which is the classic
    UNFAVORABLE case for naive weight-averaging (independently-initialized same-architecture nets
    generically live in different permutation-related basins; this project's own tight cross-seed
    success-rate similarity is behavioral, not weight-space, evidence and should NOT be read as
    basin-compatibility). This gate tests it directly instead of assuming either way.

    For alpha in {0, 0.25, 0.5, 0.75, 1}: build theta = (1-alpha)*ckpt_a + alpha*ckpt_b (elementwise
    over every floating-point tensor; non-float buffers copied from ckpt_a), load into a fresh model,
    reset every RationalNorm's `initialized` buffer to False so the FIRST forward pass directly sets
    `running_ms` from the batch statistic (not a slow ~0.99-momentum EMA drift, which would need ~300
    steps to converge -- resetting `initialized` gives correct calibration from one batch), recalibrate
    with recal_batch real cached frames in train() mode (no_grad, no backward), then measure
    teacher-forced action MSE (via the model's own target_actions loss path, identical to every other
    training loop in this codebase) on a separate eval_batch of real cached frames.

    Pre-registered decision rule: GO (weight-souping is plausible, worth a real closed-loop screening-
    slice eval) if MSE at alpha=0.5 is within `barrier_ratio`x of the BETTER endpoint's MSE (no sharp
    loss barrier). NO-GO (a real barrier -- move straight to the distillation fallback instead) if the
    midpoint MSE exceeds that ratio. Both outcomes are informative and cheap; NO-GO is the a-priori
    EXPECTED outcome per the independent-initialization argument above, not a failure of this probe."""
    _bootstrap()
    import json, pickle
    import numpy as np
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.normalization import RationalNorm

    dev = "cuda"; H = horizon
    alpha_list = [float(a) for a in str(alphas).split(",")]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"real: {len(samples)} samples, d_a={d_a} state_dim={state_dim}")

    tasks = {}
    from huggingface_hub import hf_hub_download, list_repo_files
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    def to_tensors(idxs):
        imgs = torch.tensor(np.stack([samples[i][0] for i in idxs])).permute(0, 3, 1, 2).float().div(255).to(dev)
        instr = torch.tensor([encode(tasks.get(samples[i][1], "")) for i in idxs], device=dev)
        states = torch.tensor((np.stack([samples[i][2] for i in idxs]) - s_mu) / s_sd, dtype=torch.float32, device=dev)
        actions = torch.tensor((np.stack([samples[i][3] for i in idxs]) - a_mu) / a_sd, dtype=torch.float32, device=dev)
        return imgs, instr, states, actions

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(samples))
    recal_idxs = perm[:min(recal_batch, len(samples))]
    eval_idxs = perm[recal_batch:recal_batch + min(eval_batch, len(samples) - recal_batch)]
    recal_imgs, recal_instr, recal_states, recal_actions = to_tensors(recal_idxs)
    eval_imgs, eval_instr, eval_states, eval_actions = to_tensors(eval_idxs)
    print(f"recal_batch={len(recal_idxs)} eval_batch={len(eval_idxs)} (disjoint, fixed seed)")

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)

    sd_a = torch.load(f"{VOL_PATH}/{ckpt_a}", map_location=dev, weights_only=True)
    sd_b = torch.load(f"{VOL_PATH}/{ckpt_b}", map_location=dev, weights_only=True)
    assert set(sd_a.keys()) == set(sd_b.keys()), "checkpoint key mismatch -- architectures differ, cannot interpolate"

    model = ChiVLA(cfg).to(dev)
    results = []
    for alpha in alpha_list:
        sd_interp = {}
        for k in sd_a:
            va, vb = sd_a[k], sd_b[k]
            if torch.is_floating_point(va):
                sd_interp[k] = (1.0 - alpha) * va + alpha * vb
            else:
                sd_interp[k] = va.clone()
        model.load_state_dict(sd_interp)
        for m in model.modules():
            if isinstance(m, RationalNorm):
                m.initialized.fill_(False)
        model.train()
        with torch.no_grad():
            model(recal_imgs, recal_instr, recal_states,
                 torch.zeros(len(recal_idxs), dtype=torch.long, device=dev))
        model.eval()
        with torch.no_grad():
            _, mse = model(eval_imgs, eval_instr, eval_states,
                          torch.zeros(len(eval_idxs), dtype=torch.long, device=dev),
                          target_actions=eval_actions, progress=1.0)
        mse_val = float(mse.item())
        results.append({"alpha": alpha, "teacher_forced_mse": round(mse_val, 6)})
        print(f"[alpha={alpha}] teacher_forced_mse={mse_val:.6f}")

    endpoint_mses = [r["teacher_forced_mse"] for r in results if r["alpha"] in (0.0, 1.0)]
    mid = next((r for r in results if abs(r["alpha"] - 0.5) < 1e-9), None)
    better_endpoint = min(endpoint_mses) if endpoint_mses else None
    if mid is not None and better_endpoint is not None:
        ratio = mid["teacher_forced_mse"] / max(better_endpoint, 1e-12)
        go_no_go = ("GO -- run a real closed-loop screening-slice eval of the souped model"
                   if ratio <= barrier_ratio else
                   "NO-GO -- real loss barrier detected, move to the distillation fallback instead")
    else:
        ratio = None
        go_no_go = "INCONCLUSIVE -- alpha=0.5 or an endpoint missing from `alphas`"

    result = {"config": {"ckpt_a": ckpt_a, "ckpt_b": ckpt_b, "alphas": alpha_list,
                        "recal_batch": len(recal_idxs), "eval_batch": len(eval_idxs),
                        "barrier_ratio": barrier_ratio},
             "curve": results, "better_endpoint_mse": better_endpoint,
             "midpoint_to_better_endpoint_ratio": round(ratio, 4) if ratio is not None else None,
             "go_no_go": go_no_go,
             "honest_note": ("NO-GO is the a-priori expected outcome, not a failure of this probe: "
                             "libero_rollout_head constructs a fresh ChiVLA per seed with its own "
                             "torch.manual_seed, so independently-initialized same-architecture "
                             "networks generically live in different permutation-related basins. "
                             "This project's tight cross-seed success-rate similarity (0.7-0.8pp std) "
                             "is BEHAVIORAL evidence, not weight-space evidence, and should not be "
                             "read as basin-compatibility -- this gate is the actual test.")}
    json.dump(result, open(f"{VOL_PATH}/weight_soup_interpolation_gate.json", "w"), indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=2 * 3600)
def ensemble_dagger_gen_corrective(ckpts: str = "ckpt_linear_rat_conv_s0.pt,ckpt_linear_rat_conv_s1_matched.pt,ckpt_linear_rat_conv_s2_matched.pt",
                                   vision_encoder: str = "conv", norm: str = "rational",
                                   pilot_tasks: str = "3,5", eps_per_task_pre: int = 20,
                                   max_corrections_per_task: int = 15, max_steps_eval: int = 280,
                                   num_steps_wait: int = 10, res: int = 64, horizon: int = 8,
                                   n_frames: int = 100000, kp: float = 8.0, hover: float = 0.10,
                                   grasp_z_offset: float = 0.0, grasp_dwell: int = 15, lift: float = 0.15,
                                   drop_offset: float = 0.08, release_dwell: int = 10, phase_timeout: int = 60,
                                   blend_window: int = 18, max_jump_accept: float = 0.6,
                                   grasp_lift_thresh: float = 0.05, tag: str = "_ensemble_dagger"):
    """Stage A+B of the combine_ensemble_dagger lens (post-win-scaleup-planning workflow, ranked #2):
    mine failures of the ENSEMBLE POLICY specifically (elementwise-averaged 3-seed predictions, the
    same mechanism as ensemble_consensus_pilot's +6.1pp win), not a single checkpoint's failures --
    a stricter, higher-confidence trigger for "this is a genuinely hard state," then generate
    corrective demos for exactly those (task, episode) pairs using the identical blend-fixed
    P-controller + quality gate as dagger_gen_corrective (copied verbatim, not reimplemented).

    SCOPE NOTE (deliberate simplification vs. the lens's full proposal): this pass logs per-episode
    success bits for each of the 3 individual seeds AND the ensemble (4 conditions x eps_per_task_pre
    episodes, same canonical init state per episode across all 4 -- one env per task, reused, per
    this session's proven-safe pattern), and generates corrections ONLY for ensemble-fail episodes
    (Tier 1). It does NOT yet compute the lens's proposed per-step cross-model disagreement /
    Tier-2 canary set -- that is deferred as a cheap follow-on instrumentation, not dropped, once
    Tier-1 mining proves out. Runs at the SAME 2-task pilot scale as today's online_dagger win
    (bbq_sauce, tomato_sauce) per the synthesis's explicit recommendation, since this stacks two
    not-fully-understood-collapse-adjacent mechanisms (DAgger dosage + ensemble homogenization) and
    should be de-risked at small scale before any 10-task run.

    Saves {per_episode_bits (s0/s1/s2/ensemble success per task,ep), corr_frames, corr_diag} to a
    pickle for dagger_finetune_eval to consume 3x (ckpt_in=s0/s1/s2, SAME corrective_pkl) -- Stage C
    of the lens's design. Stage D (re-ensemble the 3 dagger'd checkpoints + eval against the 93.3%
    baseline and the 96.0% single-seed-dagger'd result) is a follow-up once Stage C's 3 checkpoints
    exist, not run here."""
    _bootstrap()
    import json, os, re, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig

    dev = "cuda"; H = horizon
    ckpt_list = [c.strip() for c in str(ckpts).split(",")]
    pilot_task_idxs = [int(x) for x in str(pilot_tasks).split(",")]
    name = "lerobot/libero_object_image"

    def readable(obj_body):
        return re.sub(r"_\d+$", "", obj_body).replace("_", " ")

    def obj_filter(obs):
        return {k.rsplit("_pos", 1)[0]: obs[k] for k in obs
               if k.endswith("_pos") and not k.startswith("robot0") and "basket" not in k
               and "_to_" not in k and "eef" not in k}

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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out
    real_samples = build_samples(frames)
    d_a = real_samples[0][3].shape[1]; state_dim = real_samples[0][2].shape[0]
    A = np.stack([s[3] for s in real_samples]); S = np.stack([s[2] for s in real_samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)
    print(f"real: {len(real_samples)} samples (stats only, no bulk image tensor built), d_a={d_a} state_dim={state_dim}")
    del real_samples, frames

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)
    models = []
    for c in ckpt_list:
        m = ChiVLA(cfg).to(dev)
        m.load_state_dict(torch.load(f"{VOL_PATH}/{c}", map_location=dev, weights_only=True))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} checkpoints: {ckpt_list}")

    def build_state_full(obs):
        from robosuite.utils.transform_utils import quat2axisangle
        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)

    def build_state(obs):
        v = build_state_full(obs)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        from PIL import Image
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    # libero/robosuite imported here (see dagger_finetune_eval's docstring for why import
    # location relative to heavy GPU work matters) -- this function never does bulk image-tensor
    # conversion, only small per-step tensors alongside envs, the pattern proven safe all session.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]

    @torch.no_grad()
    def rollout(env, task, init_states, ep, mode, model_idx=None):
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env.seed(ep); obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
        ok = False; t = 0
        while t < max_steps_eval and not ok:
            im, st = obs_tensors(obs)
            if mode == "single":
                a, _ = models[model_idx](im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
            else:
                chunks = []
                for m in models:
                    a, _ = m(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
                    chunks.append((a[0] * a_sd_t + a_mu_t).cpu().numpy())
                chunk = np.mean(chunks, axis=0)
            for k in range(min(8, H)):
                act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                obs, r, done, info = env.step(act.tolist()); t += 1
                if r > 0: ok = True
                if done or ok or t >= max_steps_eval: break
        return ok

    per_episode_bits = {}
    ensemble_fail_eps = {}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        bits = {"s0": [], "s1": [], "s2": [], "ensemble": []}
        efails = []
        for ep in range(eps_per_task_pre):
            for i in range(len(models)):
                bits[f"s{i}"].append(bool(rollout(env, task, init_states, ep, "single", i)))
            e_ok = bool(rollout(env, task, init_states, ep, "ensemble"))
            bits["ensemble"].append(e_ok)
            if not e_ok: efails.append(ep)
        env.close()
        per_episode_bits[ti] = bits
        ensemble_fail_eps[ti] = efails
        print(f"[task {ti}] {task.language}: s0={sum(bits['s0'])}/{eps_per_task_pre} "
             f"s1={sum(bits['s1'])}/{eps_per_task_pre} s2={sum(bits['s2'])}/{eps_per_task_pre} "
             f"ensemble={sum(bits['ensemble'])}/{eps_per_task_pre} ensemble_fails={efails}")

    corr_frames = []
    corr_diag = {"attempted": 0, "kept": 0, "rejected_jump": 0, "rejected_shove": 0, "rejected_no_obj": 0}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        eps_to_fix = ensemble_fail_eps.get(ti, [])[:max_corrections_per_task]
        if not eps_to_fix:
            continue
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        for ep in eps_to_fix:
            corr_diag["attempted"] += 1
            env.seed(ep); obs = env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)

            obj_pos0 = obj_filter(obs)
            basket_key = next((k for k in obs if "basket" in k and k.endswith("_pos")), None)
            chosen = next((k for k in obj_pos0 if readable(k) in task.language.lower()), None)
            if chosen is None or basket_key is None:
                corr_diag["rejected_no_obj"] += 1; continue
            obj_z0 = float(obj_pos0[chosen][2]); grasp_target_z = obj_z0
            basket_xy = np.asarray(obs[basket_key][:2]); basket_z = float(obs[basket_key][2])

            ep_frames = []; max_lift = -1e9; max_jump = 0.0; prev_delta = None
            phase = "hover"; phase_t = 0; t = 0
            blend_from = None; blend_step = blend_window
            episode_index = 3_000_000 + ti * 10000 + ep
            task_index = 3000 + ti
            while t < 250:
                obj_pos_t = obj_filter(obs)
                p_obj = np.asarray(obj_pos_t.get(chosen, obj_pos0[chosen]))
                p_eef = np.asarray(obs["robot0_eef_pos"])
                max_lift = max(max_lift, float(p_obj[2]) - obj_z0)

                phase_before = phase
                if phase == "hover":
                    target_raw = p_obj + np.array([0.0, 0.0, hover]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "descend", 0
                elif phase == "descend":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = -1.0
                    if np.linalg.norm(target_raw - p_eef) < 0.02 or phase_t >= phase_timeout: phase, phase_t = "grasp", 0
                elif phase == "grasp":
                    target_raw = p_obj + np.array([0.0, 0.0, grasp_z_offset]); grip = 1.0
                    if phase_t >= grasp_dwell: phase, phase_t = "lift", 0
                elif phase == "lift":
                    target_raw = np.array([p_obj[0], p_obj[1], grasp_target_z + lift]); grip = 1.0
                    if abs(p_eef[2] - (grasp_target_z + lift)) < 0.03 or phase_t >= phase_timeout: phase, phase_t = "transit", 0
                elif phase == "transit":
                    target_raw = np.array([basket_xy[0], basket_xy[1], grasp_target_z + lift]); grip = 1.0
                    if np.linalg.norm(target_raw[:2] - p_eef[:2]) < 0.04 or phase_t >= phase_timeout: phase, phase_t = "drop_descend", 0
                elif phase == "drop_descend":
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = 1.0
                    if abs(p_eef[2] - (basket_z + drop_offset)) < 0.035 or phase_t >= phase_timeout: phase, phase_t = "release", 0
                else:
                    target_raw = np.array([basket_xy[0], basket_xy[1], basket_z + drop_offset]); grip = -1.0

                if phase != phase_before:
                    blend_from = target_raw.copy(); blend_step = 0
                if blend_step < blend_window:
                    alpha = (blend_step + 1) / blend_window
                    target = blend_from + alpha * (target_raw - blend_from)
                    blend_step += 1
                else:
                    target = target_raw

                delta = np.clip(kp * (target - p_eef), -1.0, 1.0)
                if prev_delta is not None:
                    max_jump = max(max_jump, float(np.linalg.norm(delta - prev_delta)))
                prev_delta = delta

                action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip], dtype=np.float32)
                from PIL import Image
                img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)), dtype=np.uint8)
                state = build_state_full(obs)
                ep_frames.append((episode_index, t, img, state, action, task_index))
                obs, r, done, info = env.step(action.tolist())
                phase_t += 1; t += 1
                if phase == "release" and phase_t > release_dwell: break

            final_obj = obj_filter(obs).get(chosen, p_obj)
            final_xy_d = float(np.linalg.norm(np.asarray(final_obj[:2]) - basket_xy))
            final_z = float(final_obj[2])
            geo_success = final_xy_d < 0.12 and final_z < 0.15
            is_shove = geo_success and max_lift < grasp_lift_thresh

            if not geo_success:
                corr_diag["rejected_shove"] += 1 if is_shove else 0
                continue
            if is_shove:
                corr_diag["rejected_shove"] += 1; continue
            if max_jump > max_jump_accept:
                corr_diag["rejected_jump"] += 1; continue
            corr_frames.extend(ep_frames)
            corr_diag["kept"] += 1
        env.close()
    print("CORRECTIVE DEMO GENERATION (ensemble-fail-triggered):", json.dumps(corr_diag, indent=2))

    out_pkl = f"dagger_corrective{tag}.pkl"
    payload = {"per_episode_bits": per_episode_bits, "ensemble_fail_eps": ensemble_fail_eps,
              "corr_frames": corr_frames, "corr_diag": corr_diag, "pilot_task_idxs": pilot_task_idxs,
              "a_mu": a_mu, "a_sd": a_sd, "s_mu": s_mu, "s_sd": s_sd,
              "d_a": d_a, "state_dim": state_dim,
              "task_languages": {ti: suite.get_task(ti).language for ti in pilot_task_idxs},
              "ensemble_overall": round(float(np.mean([np.mean(per_episode_bits[ti]["ensemble"]) for ti in pilot_task_idxs])), 4),
              "single_seed_overalls": {f"s{i}": round(float(np.mean([np.mean(per_episode_bits[ti][f"s{i}"]) for ti in pilot_task_idxs])), 4)
                                       for i in range(len(models))}}
    pickle.dump(payload, open(f"{VOL_PATH}/{out_pkl}", "wb"))
    vol.commit()
    print(f"saved -> {out_pkl} (n_corr_frames={len(corr_frames)}, n_kept_episodes={corr_diag['kept']}, "
         f"ensemble_overall={payload['ensemble_overall']})")
    return {"per_episode_bits": per_episode_bits, "corrective_demo_stats": corr_diag,
           "ensemble_overall": payload["ensemble_overall"],
           "single_seed_overalls": payload["single_seed_overalls"], "out_pkl": out_pkl}


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def ensemble_distillation_finetune(ckpt_in: str = "ckpt_linear_rat_conv_s0.pt",
                                   teacher_ckpts: str = "ckpt_linear_rat_conv_s0.pt,ckpt_linear_rat_conv_s1_matched.pt,ckpt_linear_rat_conv_s2_matched.pt",
                                   vision_encoder: str = "conv", norm: str = "rational",
                                   n_frames: int = 100000, res: int = 64, horizon: int = 8,
                                   finetune_steps: int = 3000, lr: float = 3e-5, batch: int = 256,
                                   teacher_batch: int = 512, eval_pilot_tasks: str = "0,3,5",
                                   eval_eps_per_task: int = 20, num_steps_wait: int = 10,
                                   max_steps: int = 280, exec_h: int = 8,
                                   ckpt_out: str = "ckpt_ensemble_distilled.pt"):
    """The distillation fallback recommended by the ensemble_distillation lens, queued immediately
    after weight_soup_interpolation_gate came back a clean, decisively-predicted NO-GO (a 6998x loss
    barrier at alpha=0.5 -- the 3 conv+rational seeds are NOT linearly mode-connected, confirming the
    independent-initialization argument rather than refuting it). Converts today's ensemble win
    (elementwise-averaging the 3 seeds, +6.1pp on a 3-task screening slice) into a supervised
    regression TARGET instead of an inference-time cost: precompute all 3 teacher checkpoints'
    (frozen, eval-mode) normalized action predictions on every real cached frame, average them
    (identical mechanism to ensemble_consensus_pilot's rollout, just applied offline over the whole
    dataset instead of per-episode), then continue `ckpt_in` (s0, not a fresh init -- cheaper and
    avoids re-deriving the whole policy from scratch) with an ordinary MSE fine-tune against that
    precomputed ensemble-average target. Zero new loss types, zero new RL/reward machinery -- same
    linear-head MSE this codebase already uses everywhere, same low-LR/short-step discipline proven
    safe by today's online_dagger win (2000 steps, 5e-5) modulo the higher step count since this
    trains on the FULL real cache, not ~1100 corrective frames.

    After training, runs a closed-loop eval of the DISTILLED STUDENT ALONE (single-model inference,
    no 3x ensemble cost) on the SAME 3-task x 20-episode screening slice already recorded in
    ensemble_consensus_pilot.json (tasks 0/3/5, canonical init states) for a DIRECT, paired
    comparison against the already-verified 87.2% (mean of 3 single seeds) and 93.3% (elementwise-
    averaging ensemble) numbers on that identical slice. Go/no-go: the distilled student should
    beat 87.2% (the point of distilling at all) and ideally approach 93.3% while paying only 1x
    inference cost -- falling back to (or short of) 87.2% would mean distillation failed to transfer
    the ensemble's benefit, not just failed to fully match it."""
    _bootstrap()
    import json, os, pickle
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"; H = horizon
    teacher_ckpt_list = [c.strip() for c in str(teacher_ckpts).split(",")]
    name = "lerobot/libero_object_image"

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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    cache = f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl"
    frames = pickle.load(open(cache, "rb"))
    from collections import defaultdict
    def build_samples(fr):
        eps = defaultdict(list)
        for f in fr: eps[f[0]].append(f)
        out = []
        for ep, fs in eps.items():
            fs.sort(key=lambda z: z[1])
            for i in range(len(fs) - H):
                out.append((fs[i][2], fs[i][5], fs[i][3], np.stack([fs[i + k][4] for k in range(H)])))
        return out
    real_samples = build_samples(frames)
    d_a = real_samples[0][3].shape[1]; state_dim = real_samples[0][2].shape[0]
    A = np.stack([s[3] for s in real_samples]); S = np.stack([s[2] for s in real_samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"real: {len(real_samples)} samples, d_a={d_a} state_dim={state_dim}")

    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim, n_embodiments=1,
                    dim=384, n_layers=8, n_heads=12, action_horizon=H, action_dim=d_a,
                    action_head="linear", norm=norm, qk_norm=norm, vision_encoder=vision_encoder)

    # ---- ALL bulk GPU image-tensor conversion happens HERE, before any LIBERO env exists in this
    # process (see dagger_finetune_eval's docstring for why this ordering is load-bearing). ----
    real_imgs = torch.tensor(np.stack([s[0] for s in real_samples])).permute(0, 3, 1, 2).float().div(255).to(dev)
    real_instr = torch.tensor([encode(tasks.get(s[1], "")) for s in real_samples], device=dev)
    real_states = torch.tensor((np.stack([s[2] for s in real_samples]) - s_mu) / s_sd, dtype=torch.float32, device=dev)

    teachers = []
    for c in teacher_ckpt_list:
        m = ChiVLA(cfg).to(dev)
        m.load_state_dict(torch.load(f"{VOL_PATH}/{c}", map_location=dev, weights_only=True))
        m.eval()
        teachers.append(m)
    print(f"loaded {len(teachers)} teacher checkpoints: {teacher_ckpt_list}")

    n = len(real_samples)
    teacher_targets = torch.zeros(n, H, d_a, device=dev)
    embodiment_zeros_tb = torch.zeros(teacher_batch, dtype=torch.long, device=dev)
    with torch.no_grad():
        for i0 in range(0, n, teacher_batch):
            i1 = min(i0 + teacher_batch, n)
            bsz = i1 - i0
            eids = embodiment_zeros_tb[:bsz]
            preds = []
            for m in teachers:
                a, _ = m(real_imgs[i0:i1], real_instr[i0:i1], real_states[i0:i1], eids)
                preds.append(a)
            teacher_targets[i0:i1] = torch.stack(preds, dim=0).mean(dim=0)
            if i0 % (teacher_batch * 20) == 0: print(f"  teacher-label batch {i0}/{n}")
    print(f"precomputed ensemble-average teacher targets for all {n} real samples")
    del teachers
    torch.cuda.empty_cache()

    student = ChiVLA(cfg).to(dev)
    student.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_in}", map_location=dev, weights_only=True))
    print(f"loaded student init {ckpt_in}, distilling toward ensemble-average targets")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    tcfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=finetune_steps, warmup_frac=0.05)
    student.train()
    for step in range(finetune_steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tcfg)
        idx = torch.randint(n, (batch,), device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, _ = student(real_imgs[idx], real_instr[idx], real_states[idx],
                              torch.zeros(batch, dtype=torch.long, device=dev))
            loss = torch.nn.functional.mse_loss(pred.float(), teacher_targets[idx])
        loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step % 500 == 0: print(f"  [distill] step {step} loss {loss.item():.6f}")
    student.eval()
    torch.save(student.state_dict(), f"{VOL_PATH}/{ckpt_out}")
    del real_imgs, real_instr, real_states, teacher_targets
    torch.cuda.empty_cache()

    # ---- closed-loop eval of the distilled student alone: libero/robosuite imported HERE for the
    # first time in this process, env creation only after all bulk tensor work is done. ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image

    a_mu_t = torch.tensor(a_mu, device=dev); a_sd_t = torch.tensor(a_sd, device=dev)
    s_mu_t = torch.tensor(s_mu, dtype=torch.float32); s_sd_t = torch.tensor(s_sd, dtype=torch.float32)

    def build_state(obs):
        v = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return v[:state_dim] if len(v) >= state_dim else np.pad(v, (0, state_dim - len(v)))

    def obs_tensors(obs):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        st = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        return im, st

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    close_sign = 1.0
    DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -close_sign]
    pilot_task_idxs = [int(x) for x in str(eval_pilot_tasks).split(",")]
    if (pilot_task_idxs != [0, 3, 5] or eval_eps_per_task != 20 or max_steps != 280
            or exec_h != 8 or num_steps_wait != 10):
        raise ValueError(
            "The stored screening baselines are valid only for tasks 0,3,5 with 20 episodes, "
            "a 280-step cap, eight-action execution, and ten settling actions."
        )

    @torch.no_grad()
    def rollout_episode(env, task, init_states, ep):
        instr_ids = torch.tensor([encode(task.language)], device=dev)
        env.seed(ep); obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(num_steps_wait): obs, _, _, _ = env.step(DUMMY)
        ok = False; t = 0
        while t < max_steps and not ok:
            im, st = obs_tensors(obs)
            a, _ = student(im, instr_ids, st, torch.zeros(1, dtype=torch.long, device=dev))
            chunk = (a[0] * a_sd_t + a_mu_t).cpu().numpy()
            for k in range(min(exec_h, H)):
                act = chunk[k].copy(); act[-1] = close_sign if act[-1] > 0 else -close_sign
                obs, r, done, info = env.step(act.tolist()); t += 1
                if r > 0: ok = True
                if done or ok or t >= max_steps: break
        return ok

    per_task = {}
    for ti in pilot_task_idxs:
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = _libero_init_states_or_raise(suite, ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        succ = 0
        for ep in range(eval_eps_per_task):
            succ += int(rollout_episode(env, task, init_states, ep))
        env.close()
        per_task[ti] = round(succ / eval_eps_per_task, 3)
        print(f"[distilled student task {ti}] {task.language}: {succ}/{eval_eps_per_task}")
    overall = round(float(np.mean(list(per_task.values()))), 4)

    result = {"config": {"ckpt_in": ckpt_in, "teacher_ckpts": teacher_ckpt_list,
                        "finetune_steps": finetune_steps, "lr": lr,
                        "eval_pilot_tasks": pilot_task_idxs, "eval_eps_per_task": eval_eps_per_task},
             "distilled_student_per_task": per_task, "distilled_student_overall": overall,
             "baseline_mean_of_single_seeds": 0.8722, "baseline_ensemble": 0.9333,
             "go_no_go": ("GO -- distillation transferred the ensemble benefit" if overall > 0.8722
                         else "NO-GO -- distilled student did not even beat the mean single-seed baseline"),
             "honest_note": ("Compared on the IDENTICAL 3-task x 20-episode screening slice already "
                             "recorded in ensemble_consensus_pilot.json for a fair paired comparison, "
                             "not a different slice or protocol.")}
    json.dump(result, open(f"{VOL_PATH}/ensemble_distillation_finetune.json", "w"), indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    return result


@app.function(image=libero_image, volumes={VOL_PATH: vol}, timeout=1800)
def libero_suite_preflight(
    suites: str = "libero_object,libero_goal,libero_spatial,libero_10",
    out_name: str = "libero_suite_preflight.json",
):
    """Fail-loud audit of the official LIBERO benchmark objects before broader runs.

    This is the first implementation hook for the cross-suite capability study.  It does
    not train or evaluate a policy.  It verifies that every requested suite exists, records
    its task text and BDDL file, and checks that official canonical initial states can be
    loaded for every task.  A missing suite, BDDL file, or initial-state set is surfaced in
    the JSON and makes ``all_requested_ready`` false instead of silently falling back to
    random resets.

    ``suites`` is comma-separated because Modal's CLI handles strings more reliably than
    list-valued parameters.  Example:

        modal run modal_app.py::libero_suite_preflight \
          --suites libero_object,libero_goal,libero_spatial,libero_10
    """
    _bootstrap()
    import json
    import os
    from libero.libero import benchmark, get_libero_path

    available = benchmark.get_benchmark_dict()
    requested = [name.strip() for name in str(suites).split(",") if name.strip()]
    result = {
        "requested_suites": requested,
        "available_benchmarks": sorted(available),
        "suites": {},
    }

    for suite_name in requested:
        suite_result = {
            "available": suite_name in available,
            "n_tasks": 0,
            "canonical_states_complete": False,
            "bddl_files_complete": False,
            "ready": False,
            "tasks": [],
            "errors": [],
        }
        result["suites"][suite_name] = suite_result
        if suite_name not in available:
            suite_result["errors"].append(
                f"Benchmark key {suite_name!r} is absent. No fallback was attempted."
            )
            continue

        try:
            suite = available[suite_name]()
            suite_result["n_tasks"] = int(suite.n_tasks)
        except Exception as exc:
            suite_result["errors"].append(f"Suite construction failed: {type(exc).__name__}: {exc}")
            continue

        state_checks = []
        bddl_checks = []
        for task_idx in range(suite.n_tasks):
            task = suite.get_task(task_idx)
            bddl_path = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            task_result = {
                "task_index": task_idx,
                "language": str(task.language),
                "problem_folder": str(task.problem_folder),
                "bddl_file": str(task.bddl_file),
                "bddl_exists": os.path.isfile(bddl_path),
                "canonical_state_count": 0,
                "canonical_states_loaded": False,
            }
            bddl_checks.append(task_result["bddl_exists"])
            if not task_result["bddl_exists"]:
                suite_result["errors"].append(
                    f"Task {task_idx}: missing BDDL file at {bddl_path}"
                )

            try:
                init_states = _libero_init_states_or_raise(suite, task_idx)
                task_result["canonical_state_count"] = int(len(init_states))
                task_result["canonical_states_loaded"] = len(init_states) > 0
                if len(init_states) == 0:
                    suite_result["errors"].append(
                        f"Task {task_idx}: canonical initial-state set is empty."
                    )
            except Exception as exc:
                suite_result["errors"].append(
                    f"Task {task_idx}: canonical initial-state load failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            state_checks.append(task_result["canonical_states_loaded"])
            suite_result["tasks"].append(task_result)

        suite_result["canonical_states_complete"] = bool(state_checks) and all(state_checks)
        suite_result["bddl_files_complete"] = bool(bddl_checks) and all(bddl_checks)
        suite_result["ready"] = (
            suite_result["n_tasks"] > 0
            and suite_result["canonical_states_complete"]
            and suite_result["bddl_files_complete"]
        )

    result["all_requested_ready"] = bool(requested) and all(
        row["ready"] for row in result["suites"].values()
    )
    result["honesty_note"] = (
        "This preflight certifies simulator benchmark metadata and official reset availability. "
        "It does not certify demonstration-dataset availability, action conventions, or a "
        "protocol-matched policy result."
    )

    out_path = os.path.join(VOL_PATH, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print("RESULT:", json.dumps(result, indent=2))
    if not result["all_requested_ready"]:
        print("PREFLIGHT FAILED: inspect suite-level errors before launching training or rollouts.")
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=5 * 3600)
def funnel_libero(steps: int = 20000, dim: int = 384, layers: int = 4, lr: float = 3e-3,
                  quad_lr_ratio: float = 0.3, const_fill: float = 10.0, batch: int = 256,
                  seed: int = 0, decomposable: bool = True, n_frames: int = 100000,
                  res: int = 64, horizon: int = 8, max_task: int = 10,
                  eps_per_task: int = 50, max_steps: int = 280, exec_h: int = 8,
                  num_steps_wait: int = 10, truncation_trials: int = 0,
                  calib_batches: int = 32, render: str = "osmesa", tag: str = "v1"):
    """Train the exactly-decomposable funnel on LIBERO-Object and measure it closed loop.

    `decomposable=True` is the real arm: frozen `scalar_rbn`, no residual, no token
    mixing, so the whole trained map exports to a homogeneous tensor network and
    canonical ODT decomposes it exactly. `decomposable=False` is the matched control:
    identical shape and parameter count with `per_token` norm and a residual stream,
    which is NOT decomposable. The difference between the two is the price of
    decomposability, which is the number the program actually needs.

    With `truncation_trials > 0` the run additionally measures closed-loop success of
    the TRUNCATED tensor network at several certified epsilon levels. That curve,
    capability against certified rank on a real robot benchmark. NOTE: the Athena run of
    this experiment returned 0/1500 closed-loop with a 0.000 dense anchor, so on that
    architecture the curve is undefined. This path is retained for a policy that has
    non-zero dense success; it is not a claim.

    Writes /vol/funnel_libero_{tag}.json.
    """
    _bootstrap()
    import json, math, os, pickle, time
    # Renderer selection MUST happen before mujoco/robosuite are imported. The image
    # sets MUJOCO_GL=egl, which fails on Modal A10G with
    # "Offscreen framebuffer is not complete, error 0x8cdd". The repo's established
    # workaround (see libero_attention_weight_surgery) is software rendering via OSMesa,
    # which works but is several times slower per episode, so the rollout budget has to
    # be sized against the measured rate rather than the Athena EGL rate.
    if render == "osmesa":
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    elif render != "egl":
        raise ValueError("render must be 'osmesa' or 'egl'")
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.chi_vla_funnel import ChiVLAFunnel, ChiVLAFunnelConfig
    from xvla.train.calibrate import calibrate_rbn_sequential
    from xvla.train.canonical_odt import (
        canonicalize_homogeneous, coefficient_error_squared, compress_homogeneous,
        export_homogeneous_network, hierarchical_tail_bound_squared,
        homogeneous_forward, odt_eigensystems, top_bases)
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"; H = horizon
    # NOTE the authoritative runner for this experiment is athena/run_funnel_libero.py.
    # That one is on a cluster with working EGL (about 5 s/episode against roughly 60 s
    # here through the OSMesa fallback) and it carries fixes this path does not: the
    # episode-level train/val split and the varying-energy collapse gate. This path
    # builds its task table from the HuggingFace parquet, which IS dataset-indexed and so
    # pairs correctly with frame[5], but it is otherwise the secondary implementation.
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3],
                            np.stack([fs[i + k2][4] for k2 in range(H)])))
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6
    print(f"{len(samples)} training chunks, d_a={d_a}, state_dim={state_dim}, vocab={len(vocab)}")

    rng = np.random.default_rng(1000 + seed)
    def pack(idx):
        img = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255)
        ins = torch.tensor(np.stack([np.asarray(encode(tasks[samples[i][1]])) for i in idx]))
        st = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd).float()
        act = torch.tensor((np.stack([samples[i][3] for i in idx]) - a_mu) / a_sd).float()
        emb = torch.zeros(len(idx), dtype=torch.long)
        return img.to(dev), ins.to(dev), st.to(dev), emb.to(dev), act.to(dev)

    torch.manual_seed(seed)
    cfg = ChiVLAFunnelConfig(
        image_size=res, in_chans=3, vocab_size=len(vocab), max_instr_len=T_len,
        state_dim=state_dim, n_embodiments=1, dim=dim, n_layers=layers,
        norm=("scalar_rbn" if decomposable else "per_token"),
        # Identical in both arms so the gap prices the normalizer, not the init.
        action_horizon=H, action_dim=d_a, const_fill=const_fill)
    model = ChiVLAFunnel(cfg).to(dev)
    print(f"decomposable={decomposable} params={model.num_params()/1e6:.2f}M "
          f"in_dim={cfg.in_dim} bond={dim+1} sum_occ={2**(layers+1)-1}")

    quad, rest = [], []
    for nm, p in model.named_parameters():
        (quad if (".left.weight" in nm or ".right.weight" in nm) else rest).append(p)
    opt = torch.optim.AdamW([{"params": quad, "lr": lr * quad_lr_ratio},
                             {"params": rest, "lr": lr}],
                            betas=(0.9, 0.95), weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    tccfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=steps, warmup_frac=0.05)
    model.train()
    curve = []
    for step in range(steps + 1):
        f = _lr_at(step, tccfg) / max(lr, 1e-12)
        for g, b in zip(opt.param_groups, base): g["lr"] = b * f
        idx = rng.choice(len(samples), size=batch, replace=False)
        img, ins, st, emb, act = pack(idx)
        opt.zero_grad(set_to_none=True)
        _, loss = model(img, ins, st, emb, target_actions=act)
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 20) == 0:
            curve.append(dict(step=step, loss=loss.item()))
            print(f"  step {step:6d} loss {loss.item():.6f}", flush=True)

    if decomposable:
        def batch_at(i):
            g = np.random.default_rng(9000 + i)
            return pack(g.choice(len(samples), size=batch, replace=False))[:4]
        rep = calibrate_rbn_sequential(model, batch_at, lambda m, b: m(*b), iters=calib_batches)
        print(f"calibrated {len(rep.sites)} sites sequentially")
    model.eval()

    # Held-out open-loop action MSE. Free, and it is the signal that says whether
    # training worked at all before any rollout budget is committed.
    ho = rng.choice(len(samples), size=min(4096, len(samples)), replace=False)
    tot = n = 0.0
    with torch.no_grad():
        for i in range(0, len(ho), 512):
            img, ins, st, emb, act = pack(ho[i:i + 512])
            a, _ = model(img, ins, st, emb)
            tot += (a - act).pow(2).sum().item(); n += act.numel()
    open_loop_mse = tot / n
    # Trivial (predict the mean) reference in the same normalised units.
    trivial = float(np.mean((np.stack([samples[i][3] for i in ho]) - a_mu) ** 2 / a_sd ** 2))
    print(f"open-loop MSE {open_loop_mse:.6f}  (trivial predictor {trivial:.6f})", flush=True)

    # ---- closed-loop harness, shared by the dense model and the truncated TNs ----
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

    def rollout(predict, n_task, per_task):
        per, succ_all, t_start = {}, 0, time.time()
        for ti in range(min(n_task, suite.n_tasks)):
            task = suite.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            init_states = _libero_init_states_or_raise(suite, ti)
            if len(init_states) < per_task:
                raise RuntimeError(f"task {ti}: {len(init_states)} states < {per_task}")
            instr_ids = torch.tensor([encode(task.language)], device=dev)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
            succ = 0
            for ep in range(per_task):
                env.seed(ti * 100 + ep)          # canonical protocol seeds before reset
                env.reset(); obs = env.set_init_state(init_states[ep])
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                # Success must LATCH, and the episode must not abort on env `done`.
                # LIBERO recomputes _check_success every step, so scoring the reward of
                # the last executed step marks a policy that succeeds mid-chunk and then
                # disturbs the object as a failure. The canonical protocol breaks the
                # instant reward > 0. (The previous form also read `r` before assignment
                # if the while body never ran.)
                ok, t = False, 0
                while not ok and t < max_steps:
                    chunk = predict(obs, instr_ids)
                    for k in range(min(exec_h, len(chunk))):
                        a = np.array(chunk[k], dtype=np.float64)
                        a[-1] = 1.0 if a[-1] > 0 else -1.0
                        obs, r, _, _ = env.step(a.tolist()); t += 1
                        if r > 0:
                            ok = True
                            break
                        if t >= max_steps: break
                succ += int(ok)
            env.close()
            per[ti] = succ / per_task; succ_all += succ
        n_eps = min(n_task, suite.n_tasks) * per_task
        elapsed = time.time() - t_start
        print(f"    [rollout] {n_eps} episodes in {elapsed:.0f}s = "
              f"{elapsed/max(n_eps,1):.2f} s/episode ({render})", flush=True)
        return succ_all / n_eps, per

    @torch.no_grad()
    def predict_dense(obs, instr_ids):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(
            obs["agentview_image"][::-1, ::-1])).resize((res, res)))
        im = torch.tensor(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        stt = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
        a, _ = model(im, instr_ids, stt, torch.zeros(1, dtype=torch.long, device=dev))
        return (a[0] * a_sd_t + a_mu_t).cpu().numpy()

    overall, per_task = rollout(predict_dense, max_task, eps_per_task)
    print(f"\nCLOSED-LOOP overall {overall:.3f}  per-task {per_task}", flush=True)

    result = dict(experiment="funnel_libero", decomposable=decomposable, seed=seed,
                  dim=dim, layers=layers, steps=steps, params=model.num_params(),
                  sum_occurrences=2 ** (layers + 1) - 1, curve=curve,
                  render=render,
                  protocol=dict(max_task=max_task, eps_per_task=eps_per_task,
                                max_steps=max_steps, exec_h=exec_h),
                  open_loop_mse=open_loop_mse, trivial_open_loop_mse=trivial,
                  closed_loop_overall=overall, closed_loop_per_task=per_task)

    # ---- exact decomposition, and capability against certified truncation ----
    if decomposable:
        cpu_model = model.to("cpu")
        net = export_homogeneous_network(cpu_model, require_frozen_rbn=True)
        idx = rng.choice(len(samples), size=256, replace=False)
        img, ins, st, emb, _ = pack(idx)
        probe = (img.cpu(), ins.cpu(), st.cpu(), emb.cpu())
        import copy as _copy
        m64 = _copy.deepcopy(cpu_model).double().eval()
        with torch.no_grad():
            x64 = cpu_model.assemble_input(*probe).double()
            ref64 = m64(probe[0].double(), probe[1], probe[2].double(), probe[3])[0].reshape(256, -1)
        exact_err = float((homogeneous_forward(net, x64) - ref64).abs().max() / ref64.abs().max())
        canon = canonicalize_homogeneous(net)
        systems = odt_eigensystems(canon.network)
        norm2 = float(systems[0][0].sum())
        total_dims = sum(int(v.shape[0]) for v, _ in systems)
        print(f"[ODT] exact to {exact_err:.3e}, isometry {max(canon.isometry_errors):.3e}, "
              f"{len(systems)} bonds, {total_dims} dims")
        result.update(exact_export_error=exact_err,
                      worst_isometry_error=max(canon.isometry_errors),
                      bond_spectra=[v.tolist() for v, _ in systems],
                      total_bond_dimensions=total_dims)

        trunc = []
        for eps in (0.003, 0.01, 0.03, 0.1):
            budget, spent, ranks = (eps ** 2) * norm2, 0.0, []
            for bond, (values, _) in enumerate(systems):
                mult = 2 ** (layers - bond); keep = int(values.shape[0])
                while keep > 1 and spent + float(values[keep - 1:].sum()) * mult <= budget:
                    keep -= 1
                spent += float(values[keep:].sum()) * mult
                ranks.append(keep)
            comp = compress_homogeneous(canon.network, top_bases(systems, tuple(ranks)))
            row = dict(eps=eps, ranks=ranks, removed=total_dims - sum(ranks),
                       removed_fraction=(total_dims - sum(ranks)) / total_dims,
                       realized_eps=math.sqrt(max(spent, 0.0) / norm2))
            if truncation_trials > 0:
                comp_dev = type(comp)(comp.embedding.to(dev).float(),
                                      tuple(c.to(dev).float() for c in comp.cores),
                                      comp.head.to(dev).float())

                @torch.no_grad()
                def predict_tn(obs, instr_ids, _c=comp_dev):
                    im_np = np.asarray(Image.fromarray(np.ascontiguousarray(
                        obs["agentview_image"][::-1, ::-1])).resize((res, res)))
                    im = torch.tensor(im_np).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
                    stt = ((torch.tensor(build_state(obs)) - s_mu_t) / s_sd_t).float().unsqueeze(0).to(dev)
                    xin = cpu_model.assemble_input(im.cpu(), instr_ids.cpu(), stt.cpu(),
                                                   torch.zeros(1, dtype=torch.long)).to(dev)
                    out = homogeneous_forward(_c, xin).reshape(H, d_a)
                    return (out * a_sd_t + a_mu_t).cpu().numpy()

                # Vary episodes-per-task, never the task count. The previous form
                # evaluated only tasks 0..4 while the dense number covered all 10, so the
                # curve was not comparable to its own zero point.
                per_t = max(1, truncation_trials // max_task)
                ov, pt = rollout(predict_tn, max_task, per_t)
                row.update(closed_loop_overall=ov, closed_loop_per_task=pt)
                print(f"  eps {eps:.3f}: remove {row['removed']}/{total_dims} "
                      f"({row['removed_fraction']:.1%}) ranks {ranks} -> closed loop {ov:.3f}",
                      flush=True)
            else:
                print(f"  eps {eps:.3f}: remove {row['removed']}/{total_dims} "
                      f"({row['removed_fraction']:.1%}) ranks {ranks}")
            trunc.append(row)
        result["truncation"] = trunc

    with open(f"{VOL_PATH}/funnel_libero_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items()
                                 if k not in ("curve", "bond_spectra")}, indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=6 * 3600)
def train_residual_free(steps: int = 20000, lr: float = 1e-3, batch: int = 256,
                        residual: bool = False, norm: str = "scalar_rbn",
                        vit_residual_pin: str = "inherit", seed: int = 0,
                        n_frames: int = 100000, res: int = 64, horizon: int = 8,
                        vision_encoder: str = "vit", calib_batches: int = 50,
                        holdout_frac: float = 0.1, ckpt_out: str = "", tag: str = ""):
    """RUNG 3: train a strict-funnel (residual-free) chi-VLA from scratch.

    This is the load-bearing empirical risk of the whole decomposability program.
    A token-wise residual stream is incompatible with a narrow ODT separator (it
    routes the full N*d layer state around every bottleneck) and it is what lifts
    the attention core's Kronecker rank from 1 to 2. NOTE the growth law is NOT
    2^S: that was refuted by measurement. For token-diagonal sublayers the rank
    stays 1, and for token-mixing ones the measured recurrence is r -> r*(r+1),
    i.e. 2, 6, 42, 1806 (doubly exponential). In the chain/bond ODT formulation it
    instead saturates, with R_i of Kronecker rank N and Q_i of rank N(N+1)/2,
    which is 5778 at the production N=107. Either way removing the residual is the
    precondition for exact global ODT, and
    nobody has ever trained this model without it. Residual connections exist
    because deep networks need them to optimise, so "algebraically sufficient"
    (the bilinear FFN carries linear and constant terms via homogeneous
    coordinates, so an identity path is representable *inside* the cores) and
    "trains to competitive success" are different claims. This measures the second.

    Run it twice with identical everything except ``residual`` so the delta
    isolates the residual's contribution rather than confounding it with the norm
    change measured in rung 2:

        train_residual_free --residual      --tag res_s0     (from-scratch control)
        train_residual_free --no-residual   --tag funnel_s0  (the funnel)

    Both default to ``norm="scalar_rbn"``, the ODT-compatible normalizer, since
    that is the target architecture. Compare against the matched from-scratch
    control at the SAME step count, never against the 40k-step 85.3% reference.

    Writes ``/vol/train_residual_free_{tag}.json`` and a checkpoint. Get the
    closed-loop number with the canonical protocol:

        libero_closedloop_capability --ckpt <ckpt_out> --norm scalar_rbn
    """
    _bootstrap()
    import json, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.norm_swap import calibrate_scalar_norms, tail_ratio_report
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"; H = horizon
    tag = tag or ("funnel" if not residual else "res") + f"_s{seed}"
    ckpt_out = ckpt_out or f"ckpt_{'funnel' if not residual else 'res'}_{norm}_s{seed}.pt"
    if vit_residual_pin not in ("inherit", "true", "false"):
        raise ValueError("vit_residual_pin must be 'inherit', 'true' or 'false'")
    vit_residual = None if vit_residual_pin == "inherit" else (vit_residual_pin == "true")

    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples, sample_eps = [], []
    for ep, fs in eps_g.items():
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3],
                            np.stack([fs[i + k2][4] for k2 in range(H)])))
            sample_eps.append(ep)
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    rng = np.random.default_rng(0)              # split fixed across arms/seeds
    ep_ids = sorted(eps_g)
    perm = rng.permutation(len(ep_ids))
    n_hold = max(1, int(holdout_frac * len(ep_ids)))
    hold_eps = {ep_ids[i] for i in perm[:n_hold]}
    tr_idx = np.array([i for i, e in enumerate(sample_eps) if e not in hold_eps])
    ho_idx = np.array([i for i, e in enumerate(sample_eps) if e in hold_eps])
    if len(tr_idx) == 0 or len(ho_idx) == 0:
        raise RuntimeError(f"degenerate split: train={len(tr_idx)} holdout={len(ho_idx)}")
    print(f"samples={len(samples)} train={len(tr_idx)} holdout={len(ho_idx)}")

    def pack(idx):
        img = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255)
        ins = torch.tensor(np.stack([np.asarray(encode(tasks[samples[i][1]])) for i in idx]))
        st = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd).float()
        act = torch.tensor((np.stack([samples[i][3] for i in idx]) - a_mu) / a_sd).float()
        emb = torch.zeros(len(idx), dtype=torch.long)
        return img.to(dev), ins.to(dev), st.to(dev), emb.to(dev), act.to(dev)

    step_rng = np.random.default_rng(1000 + seed)
    def train_batch():
        sel = step_rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
        img, ins, st, emb, _ = pack(sel)
        return (img, ins, st, emb)

    torch.manual_seed(seed)
    cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                    vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim,
                    n_embodiments=1, dim=384, n_layers=8, n_heads=12, action_horizon=H,
                    action_dim=d_a, action_head="linear", norm=norm, qk_norm=norm,
                    vision_encoder=vision_encoder, residual=residual,
                    vit_residual=vit_residual)
    model = ChiVLA(cfg).to(dev)
    print(f"residual={residual} vit_residual={vit_residual} norm={norm} "
          f"params={model.num_params()/1e6:.2f}M")

    @torch.no_grad()
    def holdout_mse(m):
        was = m.training; m.eval()
        tot = n = 0.0
        for i in range(0, len(ho_idx), 512):
            img, ins, st, emb, act = pack(ho_idx[i:i + 512])
            a, _ = m(img, ins, st, emb)
            tot += (a - act).pow(2).sum().item(); n += act.numel()
        if was: m.train()
        return tot / n

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    tccfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=steps, warmup_frac=0.05)
    model.train()
    curve, diverged = [], False
    for step in range(steps + 1):
        for g in opt.param_groups: g["lr"] = _lr_at(step, tccfg)
        sel = step_rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
        img, ins, st, emb, act = pack(sel)
        opt.zero_grad(set_to_none=True)
        _, loss = model(img, ins, st, emb, target_actions=act)
        if not torch.isfinite(loss):
            print(f"  DIVERGED at step {step}: loss={loss.item()}")
            diverged = True
            break
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 20) == 0:
            curve.append(dict(step=step, loss=loss.item(), grad_norm=float(gnorm)))
            print(f"  step {step:6d} loss {loss.item():.6f} gnorm {float(gnorm):.3f}", flush=True)

    result = dict(rung="3_residual_free", residual=residual, norm=norm, seed=seed,
                  steps=steps, lr=lr, batch=batch, diverged=diverged,
                  params_millions=model.num_params() / 1e6, curve=curve,
                  ckpt_out=ckpt_out if not diverged else None)
    if not diverged:
        calib = calibrate_scalar_norms(model, train_batch, iters=calib_batches)
        tails = tail_ratio_report(model, train_batch, iters=10)
        worst = max(((s["max_ratio"], n) for n, s in tails.items() if s["n_batches"]),
                    default=(0.0, ""))
        mse = holdout_mse(model)
        torch.save(model.state_dict(), f"{VOL_PATH}/{ckpt_out}")
        result.update(holdout_mse=mse, calibration=calib.as_dict(),
                      worst_fold_tail_ratio=worst[0], worst_fold_tail_site=worst[1],
                      next_step=(f"libero_closedloop_capability --ckpt {ckpt_out} "
                                 f"--norm {norm} --max-task 10 --eps-per-task 50 "
                                 "--max-steps 280"))
        print(f"holdout mse {mse:.6f}  worst fold tail {worst[0]:.3f}")
    with open(f"{VOL_PATH}/train_residual_free_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items() if k != "calibration"},
                                indent=2))
    return result


@app.function(image=libero_image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=4 * 3600)
def scalar_norm_refold(ckpt_in: str = "ckpt_linear_rat_vit_s0_v2.pt", ckpt_out: str = "",
                       finetune_steps: int = 2000, lr: float = 1e-4, batch: int = 256,
                       calib_batches: int = 50, n_frames: int = 100000, res: int = 64,
                       horizon: int = 8, vision_encoder: str = "vit", holdout_frac: float = 0.1,
                       tag: str = "v1"):
    """RUNG 2: what does the ODT-required normalization actually cost?

    ODT needs a polynomial map. The trained checkpoints use ``norm="rational"``,
    a genuine rational function whose mean-square is taken over the whole bond,
    so ``P/Q`` depends on the very coordinates a bond truncation discards and the
    Gram tail bounds nothing. ``canonical_odt.export_homogeneous_network`` accepts
    only ``RmsBatchNorm`` or ``Identity``. So the decomposable policy has to run on
    ``scalar_rbn``, and the cost of that has never been measured: the
    ``product_strict`` run meant to measure it never returned a result.

    This is the cheapest of the three rungs and the longest overdue. It reports
    offline MSE for three conditions on a held-out split, which is a cheap
    predictor before any rollout is spent:

        rational_baseline  the checkpoint as trained
        scalar_zeroshot    norm swapped + recalibrated, NO finetune
        scalar_finetuned   swapped + recalibrated + ``finetune_steps`` at ``lr``

    plus the fold-safety diagnostic ``max_i(perTokenRMS_i / c)`` per site. That
    ratio is near 1 exactly where a frozen scalar divide equals the per-token
    divide it replaced, so it localises the damage when the swap hurts.

    Writes ``/vol/scalar_norm_refold_{tag}.json`` and the finetuned checkpoint.
    Get the closed-loop number by then running, on the SAME canonical protocol as
    ``artifacts/vit_capability`` (10 tasks, 50 canonical eps, 280-step cap):

        libero_closedloop_capability --ckpt <ckpt_out> --norm scalar_rbn

    NOTE the zeroshot arm is expected to be poor and is NOT the headline: the
    weights were trained against a different normalizer. ``scalar_finetuned`` is
    the number that decides whether a decomposable policy is affordable.
    """
    _bootstrap()
    import json, pickle
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.train.norm_swap import (calibrate_scalar_norms, swap_norm_weights,
                                      tail_ratio_report)
    from xvla.train.train_lm import _lr_at, TrainConfig

    dev = "cuda"; H = horizon
    ckpt_out = ckpt_out or ckpt_in.replace(".pt", "_scalarrbn.pt")

    # ---- data: identical pipeline to libero_closedloop_capability -------------
    name = "lerobot/libero_object_image"
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
    T_len = 32
    def encode(s):
        ids = [1] + [vocab.get(w, 0) for w in s.lower().replace(".", "").split()]
        return (ids[:T_len] + [0] * max(0, T_len - len(ids)))[:T_len]

    frames = pickle.load(open(f"{VOL_PATH}/libero_frames_{n_frames}_{res}.pkl", "rb"))
    from collections import defaultdict
    eps_g = defaultdict(list)
    for f in frames: eps_g[f[0]].append(f)
    samples = []
    sample_eps = []          # episode id per sample, recorded HERE so the split cannot
    for ep, fs in eps_g.items():   # drift from `samples` via a second iteration pass
        fs.sort(key=lambda z: z[1])
        for i in range(len(fs) - H):
            samples.append((fs[i][2], fs[i][5], fs[i][3],
                            np.stack([fs[i + k2][4] for k2 in range(H)])))
            sample_eps.append(ep)
    assert len(sample_eps) == len(samples)
    d_a = samples[0][3].shape[1]; state_dim = samples[0][2].shape[0]
    A = np.stack([s[3] for s in samples]); S = np.stack([s[2] for s in samples])
    a_mu, a_sd = A.mean((0, 1)), A.std((0, 1)) + 1e-6
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-6

    # Episode-level split so held-out frames are not neighbours of train frames.
    rng = np.random.default_rng(0)
    ep_ids = sorted(eps_g)
    perm = rng.permutation(len(ep_ids))
    n_hold = max(1, int(holdout_frac * len(ep_ids)))
    hold_eps = {ep_ids[i] for i in perm[:n_hold]}
    tr_idx = np.array([i for i, e in enumerate(sample_eps) if e not in hold_eps])
    ho_idx = np.array([i for i, e in enumerate(sample_eps) if e in hold_eps])
    if len(tr_idx) == 0 or len(ho_idx) == 0:
        raise RuntimeError(f"degenerate split: train={len(tr_idx)} holdout={len(ho_idx)}")
    assert not (set(sample_eps[i] for i in tr_idx) & set(sample_eps[i] for i in ho_idx)), \
        "episode leaked across the train/holdout split"
    print(f"samples={len(samples)} train={len(tr_idx)} holdout={len(ho_idx)} "
          f"({n_hold}/{len(ep_ids)} episodes)")

    def pack(idx):
        img = torch.tensor(np.stack([samples[i][0] for i in idx])).permute(0, 3, 1, 2).float().div(255)
        ins = torch.tensor(np.stack([np.asarray(encode(tasks[samples[i][1]])) for i in idx]))
        st = torch.tensor((np.stack([samples[i][2] for i in idx]) - s_mu) / s_sd).float()
        act = torch.tensor((np.stack([samples[i][3] for i in idx]) - a_mu) / a_sd).float()
        emb = torch.zeros(len(idx), dtype=torch.long)
        return img.to(dev), ins.to(dev), st.to(dev), emb.to(dev), act.to(dev)

    def train_batch():
        sel = rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
        img, ins, st, emb, _ = pack(sel)
        return (img, ins, st, emb)

    def build(norm_mode):
        qk = norm_mode
        cfg = VLAConfig(image_size=res, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
                        vocab_size=len(vocab), max_instr_len=T_len, state_dim=state_dim,
                        n_embodiments=1, dim=384, n_layers=8, n_heads=12, action_horizon=H,
                        action_dim=d_a, action_head="linear", norm=norm_mode, qk_norm=qk,
                        vision_encoder=vision_encoder)
        return ChiVLA(cfg).to(dev)

    @torch.no_grad()
    def holdout_mse(model):
        model.eval()
        tot = n = 0.0
        for i in range(0, len(ho_idx), 512):
            img, ins, st, emb, act = pack(ho_idx[i:i + 512])
            a, _ = model(img, ins, st, emb)
            tot += (a - act).pow(2).sum().item(); n += act.numel()
        return tot / n

    # ---- 1. baseline, exactly as trained -------------------------------------
    src = build("rational")
    src.load_state_dict(torch.load(f"{VOL_PATH}/{ckpt_in}", map_location=dev, weights_only=True))
    src.eval()
    mse_baseline = holdout_mse(src)
    print(f"[rational_baseline] holdout mse {mse_baseline:.6f}")

    # ---- 2. swap + calibrate, no finetune ------------------------------------
    dst = build("scalar_rbn")
    swap = swap_norm_weights(dst, src.state_dict())
    print(f"[swap] transferred {swap.transferred_tensors} tensors "
          f"({swap.transferred_elements} elements), scalar sites={len(swap.scalar_norm_sites)}")
    calib = calibrate_scalar_norms(dst, train_batch, iters=calib_batches)
    if calib.unexercised:
        print(f"[swap] WARNING unexercised norm sites left uncalibrated: {calib.unexercised}")
    tails = tail_ratio_report(dst, train_batch, iters=10)
    worst = max(((s["max_ratio"], n) for n, s in tails.items() if s["n_batches"]), default=(0.0, ""))
    print(f"[swap] worst fold tail ratio {worst[0]:.3f} at {worst[1]}")
    mse_zeroshot = holdout_mse(dst)
    print(f"[scalar_zeroshot] holdout mse {mse_zeroshot:.6f} "
          f"({mse_zeroshot / mse_baseline:.2f}x baseline)")

    # ---- 3. short finetune under the new normalizer --------------------------
    mse_ft = None
    if finetune_steps > 0:
        opt = torch.optim.AdamW(dst.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        tccfg = TrainConfig(train_bin="", val_bin="", lr=lr, max_steps=finetune_steps,
                            warmup_frac=0.05)
        dst.train()
        for step in range(finetune_steps + 1):
            for g in opt.param_groups: g["lr"] = _lr_at(step, tccfg)
            sel = rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
            img, ins, st, emb, act = pack(sel)
            opt.zero_grad(set_to_none=True)
            _, loss = dst(img, ins, st, emb, target_actions=act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dst.parameters(), 1.0)
            opt.step()
            if step % max(1, finetune_steps // 10) == 0:
                print(f"  ft step {step:5d} loss {loss.item():.6f}", flush=True)
        # Recalibrate and refreeze AFTER finetuning: the constants moved with the weights.
        calib = calibrate_scalar_norms(dst, train_batch, iters=calib_batches)
        tails = tail_ratio_report(dst, train_batch, iters=10)
        worst = max(((s["max_ratio"], n) for n, s in tails.items() if s["n_batches"]),
                    default=(0.0, ""))
        mse_ft = holdout_mse(dst)
        print(f"[scalar_finetuned] holdout mse {mse_ft:.6f} "
              f"({mse_ft / mse_baseline:.2f}x baseline), worst tail {worst[0]:.3f}")

    torch.save(dst.state_dict(), f"{VOL_PATH}/{ckpt_out}")
    result = dict(
        rung="2_scalar_norm_refold", ckpt_in=ckpt_in, ckpt_out=ckpt_out,
        finetune_steps=finetune_steps, lr=lr, calib_batches=calib_batches,
        n_train_samples=len(tr_idx), n_holdout_samples=len(ho_idx),
        holdout_episodes=n_hold, total_episodes=len(ep_ids),
        holdout_mse=dict(rational_baseline=mse_baseline, scalar_zeroshot=mse_zeroshot,
                         scalar_finetuned=mse_ft),
        mse_ratio_vs_baseline=dict(
            scalar_zeroshot=mse_zeroshot / mse_baseline,
            scalar_finetuned=(mse_ft / mse_baseline) if mse_ft is not None else None),
        swap_report=swap.as_dict(), calibration=calib.as_dict(),
        fold_tail_ratios=tails, worst_fold_tail_ratio=worst[0], worst_fold_tail_site=worst[1],
        next_step=(f"libero_closedloop_capability --ckpt {ckpt_out} --norm scalar_rbn "
                   "--max-task 10 --eps-per-task 50 --max-steps 280"),
    )
    with open(f"{VOL_PATH}/scalar_norm_refold_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    vol.commit()
    print("RESULT:", json.dumps({k: v for k, v in result.items()
                                 if k not in ("fold_tail_ratios", "swap_report")}, indent=2))
    return result


@app.function(image=image, gpu="A10G", volumes={VOL_PATH: vol}, timeout=3 * 3600)
def binding_probe(steps: int = 3000, seeds: int = 3, arms: str = "", tag: str = "v1"):
    """Rung 1: can a non-attention token mixer construct the bound quantity?

    Wraps ``scripts/binding_probe_v1.py`` so the four mixing arms train on a GPU.
    See that script's module docstring for the design, the pre-registered gate, and
    how to read each outcome. Writes ``/vol/binding_probe_{tag}.json``.

    The positive control matters more than the verdict here: if the bilinear
    reference arm does not reproduce the Exp C-v2 post-attention probe R^2 (~0.90),
    the run is UNINFORMATIVE rather than negative, and must not be read as a kill
    signal for the bounded-width program.
    """
    _bootstrap()
    import json
    import sys

    sys.argv = ["binding_probe_v1", "--steps", str(steps), "--seeds", str(seeds),
                "--device", "cuda", "--out", f"{VOL_PATH}/binding_probe_{tag}.json"]
    if arms:
        sys.argv += ["--arms", arms]
    sys.path.insert(0, f"{PROJ}/scripts")
    import binding_probe_v1

    # commit in `finally`: if the summary formatting raises after the JSON is written,
    # the volume must still receive the results rather than losing the whole run.
    try:
        binding_probe_v1.main()
    finally:
        vol.commit()
    with open(f"{VOL_PATH}/binding_probe_{tag}.json") as fh:
        payload = json.load(fh)
    print("RESULT:", json.dumps(payload["gate_result"], indent=2))
    return payload["gate_result"]


@app.local_entrypoint()
def main():
    """Default: validate operators, then run the smoke ablation."""
    validate.remote()
    print("smoke ablation:", smoke_ablation.remote())
