#!/usr/bin/env python3
"""Authenticated real training-cache panel and native block bridge.

Freeze sample identities using metadata before loading the policy. This boundary
performs forwards only, with the direct-only guard installed before constructors.
It neither changes nor loads the accepted ODT decomposition. NumPy reference
replay uses an independently transcribed forward, and the exported pickle-free
arrays permit the same replay in the separately pinned reference runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np

from scripts.odt_direct_only_compliance import (
    audit_direct_only_launch, install_direct_only_runtime_guard,
    assert_direct_only_runtime_guard,
)
from research.odt_reference.block_oracle import block_forward

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
TRAINING_SHA256 = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
CACHE_RECORD_COUNT = 66_984
DATASET_TO_OFFICIAL = {0: 9, 1: 4, 2: 1, 3: 3, 4: 0, 5: 7, 6: 2, 7: 6, 8: 5, 9: 8}
OBJECTS = ("alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
           "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice")
PAIR_POSITIONS = ((27, 28), (0, 63))
PANEL_SEED = 2026090701
NATIVE_TOLERANCES = {"absolute": 2e-4, "scaled": 2e-5,
                     "scale": "max(1,max(abs(reference)))", "require_both": True}
CONFIG = dict(image_size=64, patch_size=8, vit_dim=192, vit_layers=4, vit_heads=8,
              vit_ffn_rank=576, vocab_size=26, max_instr_len=32, state_dim=8,
              n_embodiments=1, dim=384, n_layers=8, n_heads=12, ffn_rank=1152,
              action_horizon=8, action_dim=7, action_head="linear", vision_encoder="vit",
              attn="bilinear", vit_attn="bilinear", ffn="bilinear", norm="rational",
              qk_norm="rational", residual=True, vit_residual=True)
PADE_NUMERATOR = (5.3299665451049805, 7.535519123077393, 0.32727372646331787)
PADE_DENOMINATOR = (1.0, 9.648269653320312, 2.606637477874756)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError("object array is outside the pickle-free boundary")
    digest = hashlib.sha256(canonical_bytes({"dtype": value.dtype.str, "shape": list(value.shape)}))
    digest.update(value.tobytes())
    return digest.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"physical input identity mismatch: {path}")
    return path


def publish_json(path, value):
    """Write once. Existing identical frozen data is reusable, never replaced."""
    path = Path(path)
    payload = canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace an existing frozen artifact: {path}")
        return sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return sha256(path)


def panel_audit():
    result = audit_direct_only_launch(PROJECT_ROOT, (
        Path(__file__), Path(__file__).with_name("test_odt_real_panel_v1.py")),
        require_direct_qr=False)
    for field in ("prohibited_calls_found", "prohibited_self_overlap_sites",
                  "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
        if result[field]:
            raise RuntimeError(f"panel source audit failed {field}: {result[field]}")
    return result


def authenticate_sources(manifest_path, audit):
    expected = audit["source_sha256"]
    actual = {}
    for line in Path(manifest_path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in actual:
            raise RuntimeError("unsafe or repeated source-manifest path")
        require_hash(PROJECT_ROOT / relative, digest)
        actual[relative] = digest
    if actual != expected:
        raise RuntimeError("source manifest must exactly bind the audited source closure")
    return sha256(manifest_path)


def load_cache(path, provenance_path):
    """Only deserialize the byte-pinned project-owned cache, never arbitrary pickle."""
    require_hash(path, CACHE_SHA256)
    provenance = json.loads(Path(provenance_path).read_text())
    mapping = {int(k): v for k, v in provenance.get("metadata", {}).get("dataset_to_official_task", {}).items()}
    if (provenance.get("verified") is not True or provenance.get("suite") != "libero_object"
            or provenance.get("cache", {}).get("sha256") != CACHE_SHA256
            or provenance.get("cache", {}).get("frames") != CACHE_RECORD_COUNT
            or mapping != DATASET_TO_OFFICIAL
            or provenance.get("source", {}).get("split") != "train"
            or provenance.get("source", {}).get("content_hashes_match") is not True):
        raise RuntimeError("cache provenance contract differs")
    with Path(path).open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, (list, tuple)) or len(cache) != CACHE_RECORD_COUNT:
        raise RuntimeError("cache record count differs")
    return cache


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def metadata_panel(cache, *, dev_per_task=2, confirmation_per_task=8, seed=PANEL_SEED):
    """Selection depends on task/episode/frame IDs only, never pixels or outputs."""
    if dev_per_task < 1 or confirmation_per_task < 1:
        raise ValueError("both episode partitions must be nonempty")
    groups = {}
    for offset, row in enumerate(cache):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError(f"malformed cache row {offset}")
        episode = _integer(row[0], "episode")
        frame = _integer(row[1], "frame")
        dataset_task = _integer(row[5], "dataset task")
        if dataset_task not in DATASET_TO_OFFICIAL:
            raise ValueError("unknown dataset task")
        task = DATASET_TO_OFFICIAL[dataset_task]
        frames = groups.setdefault((task, episode), {})
        if frame in frames:
            raise ValueError("duplicate task/episode/frame identity")
        frames[frame] = offset
    partitions = {"development": [], "confirmation": []}
    inventory = {}
    for task in range(10):
        episodes = [ep for (t, ep), frames in groups.items()
                    if t == task and len(frames) >= 4 and max(frames) > min(frames)]
        inventory[str(task)] = len(episodes)
        if len(episodes) < dev_per_task + confirmation_per_task:
            raise ValueError(f"task {task} lacks the predeclared episode coverage")
        episodes.sort(key=lambda ep: (hashlib.sha256(f"{seed}:{task}:{ep}".encode()).hexdigest(), ep))
        for position, episode in enumerate(episodes[:dev_per_task + confirmation_per_task]):
            split = "development" if position < dev_per_task else "confirmation"
            frames = groups[(task, episode)]
            ids = sorted(frames)
            selected = (ids[(len(ids) - 1) // 4], ids[3 * (len(ids) - 1) // 4])
            if selected[1] - selected[0] < (ids[-1] - ids[0]) / 3:
                raise ValueError("quartile frames lack declared one-third episode-span separation")
            for frame_slot, frame in enumerate(selected):
                offset = frames[frame]
                partitions[split].append({
                    "task_id": task, "dataset_task_id": int(cache[offset][5]),
                    "episode_id": episode, "frame_id": frame, "frame_slot": frame_slot,
                    "cache_offset": offset,
                    "instruction": f"pick up the {OBJECTS[task]} and place it in the basket",
                })
    return {"seed": seed, "dev_per_task": dev_per_task,
            "confirmation_per_task": confirmation_per_task, "eligible_episodes_per_task": inventory,
            "selection": "per-task SHA256(seed:task:episode) order, dev first, confirmation next",
            "frames": "index quartiles floor((n-1)/4), floor(3(n-1)/4), at least 1/3 ID span",
            "pair_positions": [list(x) for x in PAIR_POSITIONS], "partitions": partitions}


def expand_pair_records(frame_records):
    return [{**record, "image_index": index, "pair_id": pair_id, "pair_positions": list(pair)}
            for index, record in enumerate(frame_records)
            for pair_id, pair in enumerate(PAIR_POSITIONS)]


def frozen_panel(cache, provenance_path):
    selected = metadata_panel(cache)
    for records in selected["partitions"].values():
        for record in records:
            row = cache[record["cache_offset"]]
            image, state, action = map(np.asarray, row[2:5])
            if image.shape != (64, 64, 3) or image.dtype != np.uint8:
                raise ValueError("cache image must be uint8 HWC64 RGB")
            if state.shape != (8,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("cache state/action is malformed")
            record["image_sha256"] = array_sha256(image)
            record["state_sha256"] = array_sha256(state)
            record["action_sha256"] = array_sha256(action)
    return {"schema": "odt-real-panel-freeze-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
            "training_sha256": TRAINING_SHA256, "cache_sha256": CACHE_SHA256,
            "cache_provenance_sha256": sha256(provenance_path), "selection": selected,
            "native_parity_tolerances": NATIVE_TOLERANCES,
            "source_scope": "real policy training-cache inputs, not unseen policy-training data",
            "preprocessing": "cached uint8 RGB without another flip/resize, float32 NCHW /255",
            "records": expand_pair_records(selected["partitions"]["confirmation"]),
            "development_records": expand_pair_records(selected["partitions"]["development"])}


def native_environment(device):
    import torch
    if np.__version__ != "1.26.4" or torch.__version__ != "2.7.1+cu126":
        raise RuntimeError(f"native runtime pins differ: numpy={np.__version__}, torch={torch.__version__}")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("requested native device unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "tf32": False, "deterministic_algorithms": True, "threads": 4,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}



def norm_inventory(state):
    """Exact frozen74-site buffer contract without a broad helper import."""
    import torch
    sites = sorted(name[:-len(".initialized")] for name in state if name.endswith(".initialized"))
    if len(sites) != 74 or "vision.norm_out" not in sites:
        raise RuntimeError("normalization site inventory differs")
    for suffix in ("running_ms", "pa", "pb"):
        actual = sorted(name[:-len(suffix)-1] for name in state if name.endswith("."+suffix))
        if actual != sites:
            raise RuntimeError("normalization buffer pairing differs")
    rows = []
    for site in sites:
        initialized, scale = state[site+".initialized"], state[site+".running_ms"]
        if (initialized.shape != () or initialized.dtype != torch.bool or scale.shape != ()
                or not bool(torch.isfinite(scale)) or float(scale) <= 0
                or bool(initialized) != (site != "vision.norm_out")):
            raise RuntimeError("normalization initialized/running-ms contract differs")
        if site == "vision.norm_out" and float(scale) != 1.0:
            raise RuntimeError("inactive normalization scale differs")
        for suffix, expected in (("pa", PADE_NUMERATOR), ("pb", PADE_DENOMINATOR)):
            value = state[site+"."+suffix]
            if value.dtype != torch.float32 or value.shape != (3,) or not torch.equal(value, torch.tensor(expected)):
                raise RuntimeError("normalization coefficient identity differs")
        rows.append({"site": site, "initialized": bool(initialized), "running_ms": float(scale)})
    return {"count": 74, "active_count": 73, "inactive_sites": ["vision.norm_out"], "sites": rows}


def load_native_model(checkpoint, device="cuda"):
    """Reusable safe policy loader. The caller must authenticate the source closure."""
    require_hash(checkpoint, CHECKPOINT_SHA256)
    install_direct_only_runtime_guard(profile="legacy87")
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig
    from xvla.nn.normalization import RationalNorm
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 540:
        raise RuntimeError("checkpoint tensor inventory differs")
    inventory = norm_inventory(state)
    config = VLAConfig(**CONFIG)
    model = ChiVLA(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if model.num_params() != 20_137_352:
        raise RuntimeError("native parameter count differs")
    for name, module in model.named_modules():
        if isinstance(module, RationalNorm):
            module.frozen = True
            if name != "vision.norm_out" and not bool(module.initialized):
                raise RuntimeError("uninitialized active rational normalization")
    assert_direct_only_runtime_guard(exact_allowed_calls=())
    return model, {"config": asdict(config), "normalization_inventory": inventory,
                   "checkpoint_sha256": CHECKPOINT_SHA256, "parameters": model.num_params()}


def load_training(path):
    require_hash(path, TRAINING_SHA256)
    training = json.loads(Path(path).read_text())
    if (training.get("cache_sha256") != CACHE_SHA256 or training.get("norm") not in {None, "rational"}
            or training.get("suite") != "libero_object" or training.get("res") != 64
            or len(training.get("vocab", {})) != 26):
        raise RuntimeError("training inference-auxiliary contract differs")
    normalization = {key: np.asarray(training["normalization"][key], dtype=np.float32)
                     for key in ("action_mean", "action_std", "state_mean", "state_std")}
    for key, value in normalization.items():
        width = 7 if key.startswith("action") else 8
        if value.shape != (width,) or not np.isfinite(value).all() or (key.endswith("std") and not np.all(value > 0)):
            raise ValueError("malformed fixed normalization")
    return training["vocab"], normalization


def encode_instruction(text, vocab):
    tokens = [1] + [vocab.get(word, 0) for word in text.lower().replace(".", "").split()]
    return (tokens[:32] + [0] * 32)[:32]


def comparison(actual, reference, tolerances=NATIVE_TOLERANCES):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("parity comparison requires equal nonempty shapes")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return {"passed": False, "finite": False, "shape": list(actual.shape)}
    error = actual - reference
    maximum = float(np.max(np.abs(error)))
    scale = max(1.0, float(np.max(np.abs(reference))))
    per_example = np.sqrt(np.mean(error.reshape(len(error), -1) ** 2, axis=1))
    return {"passed": maximum <= tolerances["absolute"] and maximum / scale <= tolerances["scaled"],
            "finite": True, "shape": list(actual.shape), "max_abs": maximum,
            "reference_max_abs_scale": scale, "scaled_max_abs": maximum / scale,
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "per_example_rmse": per_example.tolist(), "tolerances": tolerances}


def capture_partition(model, cache, records, vocab, normalization, *, device, batch_size):
    import torch
    block = model.vision.blocks.blocks[0]
    images = np.stack([cache[row["cache_offset"]][2] for row in records])
    states = np.asarray([cache[row["cache_offset"]][3] for row in records], dtype=np.float32)
    instructions = np.asarray([row["instruction"] for row in records])
    tokens = np.asarray([encode_instruction(text, vocab) for text in instructions], dtype=np.int64)
    standardized = (states - normalization["state_mean"]) / normalization["state_std"]
    frame_arrays = {"image_rgb": images, "image_instructions": instructions,
                    "image_states_raw": states, "image_states_normalized": standardized,
                    "image_token_ids": tokens,
                    "image_actions_cache": np.asarray([cache[row["cache_offset"]][4] for row in records], dtype=np.float32)}
    chunks = {key: [] for key in ("image_ingress64", "image_post_attention64", "image_output64",
                                  "image_actions_normalized", "image_actions")}
    inactive_calls = []
    def inactive_sentinel(module, inputs):
        inactive_calls.append(True)
        raise RuntimeError("inactive vision.norm_out was called")
    handle = model.vision.norm_out.register_forward_pre_hook(inactive_sentinel)
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                stop = min(start + batch_size, len(records))
                image_tensor = torch.from_numpy(images[start:stop].copy()).permute(0, 3, 1, 2).to(device).float().div(255)
                ingress = model.vision.patch(image_tensor).flatten(2).transpose(1, 2) + model.vision.pos_emb
                post_attention = ingress + block.attn_gain * block.attn(block.rbn_attn(ingress))
                output = block(ingress)
                prediction, loss = model(image_tensor, torch.from_numpy(tokens[start:stop]).to(device),
                    torch.from_numpy(standardized[start:stop]).to(device),
                    torch.zeros(stop - start, dtype=torch.long, device=device))
                if loss is not None or tuple(prediction.shape) != (stop - start, 8, 7):
                    raise RuntimeError("native policy output contract differs")
                prediction_array = prediction.cpu().numpy()
                values = (ingress.cpu().numpy(), post_attention.cpu().numpy(), output.cpu().numpy(),
                          prediction_array, prediction_array * normalization["action_std"] + normalization["action_mean"])
                for key, value in zip(chunks, values):
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"nonfinite native forward: {key}")
                    chunks[key].append(value)
    finally:
        handle.remove()
    frame_arrays.update({key: np.concatenate(values) for key, values in chunks.items()})
    for field in ("task", "episode", "frame"):
        frame_arrays[f"image_{field}_ids"] = np.asarray([row[f"{field}_id"] for row in records], dtype=np.int64)
    pair_records = expand_pair_records(records)
    pair_inputs = np.stack([frame_arrays["image_ingress64"][r["image_index"], r["pair_positions"]] for r in pair_records])
    context = np.stack([frame_arrays["image_output64"][r["image_index"], r["pair_positions"]] for r in pair_records])
    native_two = []
    with torch.inference_mode():
        for start in range(0, len(pair_inputs), batch_size):
            native_two.append(block(torch.from_numpy(pair_inputs[start:start + batch_size]).to(device)).cpu().numpy())
    native_two = np.concatenate(native_two)
    weights = {key: value.detach().cpu().numpy().copy() for key, value in block.state_dict().items()}
    reference_two = block_forward(pair_inputs, weights, n_heads=8, mask=np.ones((2, 2)))
    reference64_chunks, reference_r_chunks = [], []
    for start in range(0, len(records), batch_size):
        output, trace = block_forward(frame_arrays["image_ingress64"][start:start + batch_size], weights,
                                      n_heads=8, mask=np.ones((64, 64)), return_trace=True)
        reference64_chunks.append(output)
        reference_r_chunks.append(trace["residual1"])
    reference64 = np.concatenate(reference64_chunks)
    reference_r = np.concatenate(reference_r_chunks)
    # A fixed pair mask keeps all64 token rows but exposes the chosen two keys.
    masked = []
    with torch.inference_mode():
        for pair in PAIR_POSITIONS:
            mask = torch.zeros((64, 64), dtype=torch.float32, device=device)
            mask[:, list(pair)] = 1.0
            values = []
            for start in range(0, len(records), batch_size):
                x = torch.from_numpy(frame_arrays["image_ingress64"][start:start + batch_size]).to(device)
                values.append(block(x, mask=mask)[:, list(pair)].cpu().numpy())
            masked.append(np.concatenate(values))
    masked_rows = np.stack(masked, axis=1).reshape(native_two.shape)
    frame_arrays.update({"image_reference_output64": reference64, "image_reference_post_attention64": reference_r})
    pair_arrays = {"real_inputs": pair_inputs.astype(np.float64), "native_two_token": native_two,
                   "native_context_rows": context, "native_two_outputs": native_two,
                   "native_context_outputs": context, "reference_two_outputs": reference_two,
                   "numpy_two_token": reference_two,
                   "native_masked_context_rows": masked_rows,
                   "pair_positions": np.asarray([r["pair_positions"] for r in pair_records], dtype=np.int64),
                   "pair_ids": np.asarray([r["pair_id"] for r in pair_records], dtype=np.int64)}
    for field in ("task", "episode", "frame"):
        pair_arrays[f"{field}_ids"] = np.asarray([r[f"{field}_id"] for r in pair_records], dtype=np.int64)
    parity = {"native_two_vs_numpy": comparison(native_two, reference_two),
              "native64_vs_numpy": comparison(frame_arrays["image_output64"], reference64),
              "native_post_attention_vs_numpy": comparison(frame_arrays["image_post_attention64"], reference_r),
              "native_two_vs_pair_mask64": comparison(native_two, masked_rows),
              "context_removal_descriptive_only": comparison(native_two, context),
              "inactive_vision_norm_calls": len(inactive_calls)}
    return {**frame_arrays, **pair_arrays}, weights, parity


def run_capture(args):
    start = time.monotonic()
    audit = panel_audit()
    source_manifest_sha = authenticate_sources(args.source_manifest, audit)
    require_hash(args.checkpoint, CHECKPOINT_SHA256)
    require_hash(args.training, TRAINING_SHA256)
    cache = load_cache(args.cache, args.provenance)
    freeze = frozen_panel(cache, args.provenance)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "panel_freeze.json"
    freeze_sha = publish_json(freeze_path, freeze)
    print(json.dumps({"phase": "identities_frozen_before_model", "freeze_sha256": freeze_sha,
                      "confirmation_pairs": len(freeze["records"])}), flush=True)
    if args.freeze_only:
        return
    if (output / "panel.npz").exists() or (output / "panel_manifest.json").exists():
        raise RuntimeError("capture output already exists, refusing to replace evidence")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("unexpected deterministic CUDA workspace configuration")
    environment = native_environment(args.device)
    model, model_record = load_native_model(args.checkpoint, args.device)
    vocab, normalization = load_training(args.training)
    arrays, parity, weights = {}, {}, None
    for split, prefix in (("development", "dev_"), ("confirmation", "")):
        values, current_weights, measures = capture_partition(model, cache,
            freeze["selection"]["partitions"][split], vocab, normalization,
            device=args.device, batch_size=args.batch_size)
        arrays.update({prefix + key: value for key, value in values.items()})
        parity[split] = measures
        weights = current_weights
        print(json.dumps({"phase": "captured", "split": split,
                          "native_two_max_abs": measures["native_two_vs_numpy"].get("max_abs")}), flush=True)
        required = [v["passed"] for key, v in measures.items()
                    if key not in {"context_removal_descriptive_only", "inactive_vision_norm_calls"}]
        if split == "development" and not all(required):
            publish_json(output / "development_failure.json", {"parity": parity, "freeze_sha256": freeze_sha})
            raise RuntimeError("development native parity failed, confirmation evaluation stopped")
    arrays.update({"weight__" + key: value for key, value in weights.items()})
    arrays.update({"normalization__" + key: value for key, value in normalization.items()})
    for value in arrays.values():
        if np.asarray(value).dtype.hasobject:
            raise RuntimeError("object array rejected from output boundary")
    panel_path = output / "panel.npz"
    with panel_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    runtime_guard = assert_direct_only_runtime_guard(exact_allowed_calls=())
    authenticate_sources(args.source_manifest, audit)
    require_hash(args.checkpoint, CHECKPOINT_SHA256)
    require_hash(args.training, TRAINING_SHA256)
    require_hash(args.cache, CACHE_SHA256)
    require_hash(freeze_path, freeze_sha)
    passed = all(v["passed"] for split in parity.values() for key, v in split.items()
                 if key not in {"context_removal_descriptive_only", "inactive_vision_norm_calls"})
    manifest = {"schema": "odt-real-panel-v1", "passed": passed,
        "panel_sha256": sha256(panel_path), "freeze_sha256": freeze_sha,
        "freeze_file": freeze_path.name, "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_sha256": TRAINING_SHA256, "cache_sha256": CACHE_SHA256,
        "cache_provenance_sha256": sha256(args.provenance), "records": freeze["records"],
        "development_records": freeze["development_records"], "model": model_record,
        "block_config": {"n_heads": 8, "eps": 1e-6, "width": 192, "tokens": 2,
                         "mask": [[1, 1], [1, 1]], "causal": False},
        "native_parity_tolerances": NATIVE_TOLERANCES, "parity": parity,
        "environment": environment, "source_manifest_sha256": source_manifest_sha,
        "source_sha256": audit["source_sha256"], "runtime_guard": runtime_guard,
        "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str,
                         "sha256": array_sha256(value)} for key, value in arrays.items()},
        "source_scope": freeze["source_scope"], "preprocessing": freeze["preprocessing"],
        "canonical_odt_decomposition_performed": False, "simulator_launched": False,
        "elapsed_s": time.monotonic() - start}
    publish_json(output / "panel_manifest.json", manifest)
    print(json.dumps({"passed": passed, "panel_sha256": manifest["panel_sha256"],
                      "elapsed_s": manifest["elapsed_s"]}), flush=True)
    if not passed:
        raise RuntimeError("confirmation native bridge failed its frozen tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(panel_audit(), indent=2, sort_keys=True))
        return
    for name, value in vars(args).items():
        if name in {"source_manifest", "checkpoint", "training", "cache", "provenance", "output"} and value is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if not 1 <= args.batch_size <= 32:
        parser.error("batch size must be in [1,32]")
    run_capture(args)


if __name__ == "__main__":
    main()
