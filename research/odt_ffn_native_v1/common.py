"""Pickle-free bridge artifacts and fixed experiment identities."""
import hashlib
import json
from pathlib import Path
import numpy as np

CHECKPOINT_SHA = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
NATIVE_GATE = {"absolute": 2e-4, "scaled": 2e-5}
DOUBLE_GATE = 1e-10
DENOMINATOR_MARGIN = 1e-12
RANKS = (192, 184, 174, 154)
SEEDS = (0, 1, 2)
DEV_TOKENS = (0, 27, 63)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f"artifact identity differs: {path}")


def write_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def conditions():
    result = [{"name": "fullrank", "rank": 193, "seed": None, "construction": "leading"}]
    for rank in RANKS:
        result.append({"name": f"leading_k{rank}", "rank": rank, "seed": None, "construction": "leading"})
        for seed in SEEDS:
            result.append({"name": f"anchored_k{rank}_s{seed}", "rank": rank,
                           "seed": seed, "construction": "zero_input_anchored_direct_qr"})
    return result


def read_panel(panel, metadata):
    value = json.loads(Path(metadata).read_text())
    if value.get("passed") is not True or value.get("schema") != "odt-real-panel-v1" or value.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("native real panel did not pass its authenticated bridge")
    require_hash(panel, value["panel_sha256"])
    freeze_path = Path(metadata).parent / "panel_freeze.json"
    require_hash(freeze_path, value["freeze_sha256"])
    freeze = json.loads(freeze_path.read_text())
    if freeze["records"] != value["records"] or freeze["development_records"] != value["development_records"]:
        raise ValueError("native panel freeze differs")
    arrays = read_arrays(panel)
    for prefix, count in (("", 160), ("dev_", 40)):
        if arrays[prefix+"image_post_attention64"].shape != (count, 64, 192):
            raise ValueError("frozen native FFN ingress shape differs")
        if arrays[prefix+"image_rgb"].shape != (count, 64, 64, 3):
            raise ValueError("frozen image panel shape differs")
        tasks = arrays[prefix+"image_task_ids"]
        if tasks.dtype != np.int64 or any(int(np.sum(tasks == t)) != count//10 for t in range(10)):
            raise ValueError("native frame panel is not task balanced")
    return arrays, value


def dev_selection(arrays):
    tasks = arrays["dev_image_task_ids"]
    frames = [int(np.flatnonzero(tasks == task)[0]) for task in range(10)]
    return [{"image_index": frame, "token_index": token} for frame in frames for token in DEV_TOKENS]


def comparison(actual, expected, *, double=False):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("comparison shapes differ")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {"passed": False, "finite": False}
    maximum = float(np.max(np.abs(actual-expected)))
    denominator = max(1e-12 if double else 1., float(np.max(np.abs(expected))))
    scaled = maximum/denominator
    passed = scaled <= DOUBLE_GATE if double else maximum <= NATIVE_GATE["absolute"] and scaled <= NATIVE_GATE["scaled"]
    return {"passed": passed, "finite": True, "maximum_absolute_error": maximum,
            "scaled_error": scaled, "scale": denominator,
            "tolerance": DOUBLE_GATE if double else NATIVE_GATE}
