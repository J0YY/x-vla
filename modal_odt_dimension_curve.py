"""Isolated, pinned Modal lane for physical canonical ODT rollout experiments.

No historical Modal application is imported or copied into this image.  Initial
entrypoints only validate the environment and available CPU memory.  Production
rollouts are added only after the authentic persisted-network gate succeeds.
"""

from __future__ import annotations

import modal


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
app = modal.App("xvla-odt-dimension-curve-v1")
volume = modal.Volume.from_name("xvla-data", create_if_missing=False)

image = (
    modal.Image.from_registry("python:3.10.19-slim-bookworm")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libosmesa6-dev",
        "libegl1-mesa-dev", "libgles2-mesa-dev", "libglfw3", "libglew-dev",
        "patchelf", "gcc", "g++",
    )
    .pip_install("torch==2.7.1+cu126", "torchvision==0.22.1+cu126", extra_index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "numpy==1.26.4", "scipy==1.15.3", "pillow==12.1.1",
        "mujoco==3.5.0", "robosuite==1.4.1", "bddl==3.6.0", "easydict==1.13",
        "numba==0.64.0", "llvmlite==0.46.0", "gym==0.26.2", "PyOpenGL==3.1.10",
        "termcolor==3.3.0", "cloudpickle==3.1.2", "PyYAML==6.0.3",
        "hydra-core==1.3.2", "imageio==2.37.2", "imageio-ffmpeg==0.6.0",
        "matplotlib==3.10.8", "future==1.0.0", "h5py==3.15.1",
        "pytest==8.4.2",
    )
    .pip_install("opencv-python-headless==4.13.0.92", extra_options="--no-deps")
    .run_commands(
        "git init /opt/LIBERO",
        "git -C /opt/LIBERO remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        f"git -C /opt/LIBERO fetch --depth=1 origin {LIBERO_COMMIT}",
        f"git -C /opt/LIBERO checkout --detach {LIBERO_COMMIT}",
        "pip install --no-deps -e /opt/LIBERO",
    )
    .env({
        "PYTHONPATH": "/opt/LIBERO:/root/odt",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
    })
)


controller_image = image.add_local_file("modal_odt_dimension_curve_controller.py", "/root/odt/modal_odt_dimension_curve_controller.py").add_local_file("tests/test_modal_odt_dimension_curve_controller.py", "/root/odt/tests/test_modal_odt_dimension_curve_controller.py")


@app.function(image=controller_image, gpu="L4", cpu=(8.0, 8.0), memory=(16384, 16384), timeout=600, max_containers=1)
def environment_probe() -> dict:
    import importlib.metadata
    import hashlib
    import os
    from pathlib import Path
    import platform
    import subprocess
    import time

    import numpy as np
    import torch
    import yaml
    from modal_odt_dimension_curve_controller import install_controller, install_prohibited_route_guards, COUNTS
    numerical_guards = install_prohibited_route_guards()

    root = Path("/opt/LIBERO/libero/libero")
    configuration = {
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": "/opt/LIBERO/datasets",
    }
    destination = Path("/root/.libero/config.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(configuration))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    controller_record = install_controller()

    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "torch", "mujoco", "robosuite", "pillow", "numba", "gym",
    )}
    if versions != {
        "numpy": "1.26.4", "torch": "2.7.1+cu126", "mujoco": "3.5.0",
        "robosuite": "1.4.1", "pillow": "12.1.1", "numba": "0.64.0", "gym": "0.26.2",
    } or platform.python_version() != "3.10.19":
        raise RuntimeError(f"pinned runtime differs: {versions}")
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e":
        raise RuntimeError("official task bytes differ")
    original_load = torch.load
    def trusted_packaged_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = trusted_packaged_load
    try:
        states = suite.get_task_init_states(0)
    finally:
        torch.load = original_load
    environment = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    started = time.monotonic()
    transitions = []
    try:
        environment.seed(0)
        observation = environment.reset()
        observation = environment.set_init_state(states[0])
        for step in range(11):
            observation, reward, done, information = environment.step([0., 0., 0., 0., 0., 0., -1.])
            if type(done) not in (bool, np.bool_) or not np.isfinite(reward):
                raise RuntimeError("malformed simulator transition")
            if bool(done) or reward > 0:
                raise RuntimeError("unexpected termination during fixed renderer probe")
            transitions.append({"step": step, "done": bool(done), "reward": float(reward)})
        if observation["agentview_image"].shape != (64, 64, 3):
            raise RuntimeError("renderer image shape differs")
    finally:
        environment.close()
    return {
        "schema": "xvla_modal_odt_environment_probe_v1", "passed": True,
        "versions": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "libero_commit": subprocess.check_output(["git", "-C", "/opt/LIBERO", "rev-parse", "HEAD"], text=True).strip(),
        "transitions": transitions, "elapsed_seconds": time.monotonic() - started,
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "controller": controller_record, "controller_counts": dict(COUNTS), "numerical_guards": numerical_guards,
    }


@app.local_entrypoint()
def probe() -> None:
    import json
    print(json.dumps(environment_probe.remote(), sort_keys=True, indent=2))


@app.function(image=image, gpu="L4", cpu=(2.0, 2.0), memory=(4096, 4096), timeout=120, max_containers=1)
def build_probe() -> dict:
    import platform
    import torch
    import numpy
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": numpy.__version__, "gpu": torch.cuda.get_device_name(0)}


@app.local_entrypoint()
def build() -> None:
    print(build_probe.remote())


@app.function(image=controller_image, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=180, max_containers=1)
def controller_tests() -> dict:
    import subprocess
    result = subprocess.run(["python", "-m", "pytest", "-q", "/root/odt/tests/test_modal_odt_dimension_curve_controller.py"], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return {"passed": True, "stdout": result.stdout, "stderr": result.stderr}


@app.local_entrypoint()
def test_controller() -> None:
    print(controller_tests.remote())


SOURCE_BUNDLE_DIRECTORY = "tmp/odt_modal_curve_v1/source_v3"
SOURCE_BUNDLE_SHA256 = "f541f6fd9a784fec0cd00d428a9cd6d80084a5246393ea5cdc777158bdae6602"
worker_image = image.add_local_dir(SOURCE_BUNDLE_DIRECTORY, "/root/odt")
worker_test_image = worker_image.add_local_file("tests/test_modal_odt_dimension_curve_worker.py", "/root/odt/tests/test_modal_odt_dimension_curve_worker.py").add_local_file("tests/test_modal_odt_dimension_curve_reduced.py", "/root/odt/tests/test_modal_odt_dimension_curve_reduced.py")


def _execute_policy(removal: int, run_name: str, artifact_manifest_sha256: str = "", receipt_sha256: str = "", smoke: bool = False) -> dict:
    import contextlib
    import dataclasses
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import platform
    import resource
    import time
    import yaml

    source = Path("/root/odt")
    if hashlib.sha256((source / "bundle.json").read_bytes()).hexdigest() != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("source bundle root hash differs")
    bundle = json.loads((source / "bundle.json").read_text())
    for relative, expected in bundle["files"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source byte identity differs: {relative}")
    from modal_odt_dimension_curve_controller import install_controller, install_prohibited_route_guards, COUNTS
    for key in COUNTS:
        COUNTS[key] = 0.0 if key == "largest_relative_residual" else 0
    guards = install_prohibited_route_guards()
    from modal_odt_dimension_curve_worker import configure, load_source, load_training, read_json, run_panel, sha256

    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in run_name):
        raise ValueError("unsafe run name")
    if type(removal) is not int or removal not in (0, 30, 40, 50, 60, 70, 80):
        raise ValueError("unregistered dimension removal")
    output = Path("/vol/odt_dimension_curve_v1/results") / run_name
    output.mkdir(parents=True, exist_ok=False)
    root = Path("/opt/LIBERO/libero/libero")
    config = Path("/root/.libero/config.yaml")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"), "datasets": "/opt/LIBERO/datasets"}))
    configure()
    controller = install_controller()
    protocol = read_json(source / "protocol.json")
    inputs = Path("/vol/odt_dimension_curve_v1/inputs")
    training = load_training(inputs)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "mujoco", "robosuite", "pillow")}
    expected_versions = protocol["versions"]
    if platform.python_version() != expected_versions["python"] or any(versions[name] != expected_versions[name] for name in versions):
        raise RuntimeError(f"worker numerical environment differs: {versions}")
    started = time.monotonic()
    def publish(filename, value):
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with (output / filename).open("xb") as stream:
            stream.write(encoded)
        volume.commit()
    def progress(value):
        value.update(elapsed_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(f"progress_{value['inference_index']:03d}.json", value)
    identity = {"schema": "xvla_modal_odt_fixed20_closed_loop_v1", "requested_removal_percent": removal,
        "checkpoint_sha256": "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee",
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256, "protocol_sha256": bundle["files"]["protocol.json"],
        "controller": controller, "versions": versions, "smoke": smoke,
        "historical_baseline_is_not_the_paired_comparator": True}
    try:
        with contextlib.ExitStack() as stack:
            model = None
            mapped = None
            if removal == 0:
                model = load_source(inputs, protocol["configuration"])
            else:
                from xvla.train.implicit_projective_dag_mapped import open_mapped_implicit_projective_dag_artifact
                ladder = Path("/vol/odt_dimension_curve_v1/ladder")
                receipt_file = ladder / f"remove_{removal}.json"
                if len(receipt_sha256) != 64 or sha256(receipt_file) != receipt_sha256:
                    raise RuntimeError("dimension receipt is not authenticated")
                receipt = read_json(receipt_file)
                if receipt["artifact"]["manifest_sha256"] != artifact_manifest_sha256 or receipt["artifact"]["path"] != f"remove_{removal}":
                    raise RuntimeError("dimension receipt and artifact identity differ")
                if receipt["dimension_denominator"] != "unique_nonleaf_nonroot_tensor_output_bonds":
                    raise RuntimeError("dimension denominator differs")
                actual = 1. - receipt["retained_internal_dimensions"] / receipt["original_internal_dimensions"]
                if abs(actual - receipt["actual_internal_dimension_removal"]) > 1e-14 or abs(actual - removal / 100.) > 0.001:
                    raise RuntimeError("reported physical dimensions do not match the requested rung")
                mapped = stack.enter_context(open_mapped_implicit_projective_dag_artifact(ladder / f"remove_{removal}", expected_manifest_sha256=artifact_manifest_sha256))
                if not mapped.execution_layout_supported:
                    raise RuntimeError("artifact layout exceeds the bounded mapped executor")
                identity["dimension_receipt_sha256"] = receipt_sha256
                identity["artifact_manifest_sha256"] = artifact_manifest_sha256
                identity["actual_internal_dimension_removal"] = actual
                identity["segmented_bytes_per_evaluation"] = mapped.segmented_bytes_per_evaluation
                identity["source_model_objects_created"] = 0
            publish("started.json", identity)
            if removal == 0:
                rows = run_panel(model, mapped, training, protocol, progress, smoke=smoke)
            else:
                from modal_odt_dimension_curve_reduced import run_reduced_panel
                rows = run_reduced_panel(mapped, training, protocol, progress, smoke=smoke)
        if guards["prohibited_calls"] or guards["matrix_spectral_norm_calls"]:
            raise RuntimeError("a prohibited numerical route was attempted")
        result = {**identity, "completed": True, "episodes": rows, "successes": sum(row["success"] for row in rows),
            "episode_count": len(rows), "success_rate": sum(row["success"] for row in rows) / len(rows),
            "controller_counts": dict(COUNTS), "numerical_guards": guards,
            "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        publish("result.json", result)
        return result
    except Exception as error:
        publish("failure.json", {**identity, "completed": False, "error_type": type(error).__name__, "error": str(error),
            "controller_counts": dict(COUNTS), "numerical_guards": guards, "elapsed_seconds": time.monotonic() - started})
        raise


@app.function(image=worker_image, gpu="L4", cpu=(16.0, 16.0), memory=(32768, 32768), timeout=7200, max_containers=1, volumes={"/vol": volume})
def baseline(run_name: str, smoke: bool = False) -> dict:
    return _execute_policy(0, run_name, smoke=smoke)


@app.function(image=worker_image, gpu="L4", cpu=(16.0, 16.0), memory=(344064, 344064), timeout=43200, max_containers=6, volumes={"/vol": volume})
def reduced(removal: int, run_name: str, artifact_manifest_sha256: str, receipt_sha256: str, smoke: bool = False) -> dict:
    return _execute_policy(removal, run_name, artifact_manifest_sha256, receipt_sha256, smoke)


@app.function(image=worker_test_image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=300, max_containers=1)
def worker_tests() -> dict:
    import subprocess
    result = subprocess.run(["python", "-m", "pytest", "-q", "/root/odt/tests/test_modal_odt_dimension_curve_worker.py", "/root/odt/tests/test_modal_odt_dimension_curve_reduced.py"], capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return {"passed": True, "stdout": result.stdout, "stderr": result.stderr}


@app.local_entrypoint()
def test_worker() -> None:
    print(worker_tests.remote())


@app.local_entrypoint()
def baseline_smoke() -> None:
    print(baseline.remote("baseline_smoke_v2", smoke=True))


@app.local_entrypoint()
def evaluate(
    removal: int,
    run_name: str,
    artifact_manifest_sha256: str = "",
    receipt_sha256: str = "",
    smoke: bool = False,
) -> None:
    """Run one pinned baseline or reduced-policy panel and print its result."""
    import json

    if removal == 0:
        if artifact_manifest_sha256 or receipt_sha256:
            raise ValueError("baseline evaluation does not accept artifact hashes")
        result = baseline.remote(run_name, smoke=smoke)
    else:
        if not artifact_manifest_sha256 or not receipt_sha256:
            raise ValueError("reduced evaluation requires both artifact and receipt hashes")
        result = reduced.remote(
            removal,
            run_name,
            artifact_manifest_sha256,
            receipt_sha256,
            smoke=smoke,
        )
    print(json.dumps(result, sort_keys=True, indent=2))


@app.function(image=worker_image, cpu=(16.0, 16.0), memory=(16384, 16384), timeout=900, max_containers=1)
def mapped_benchmark(node_count: int = 10000) -> dict:
    import tempfile
    import time
    from pathlib import Path
    import resource
    import torch
    from modal_odt_dimension_curve_controller import install_prohibited_route_guards
    guards = install_prohibited_route_guards()
    from xvla.train.implicit_sparse_projective_odt import ImplicitNode, ImplicitProjectiveDAG, UnaryCore, ReducedQRBinaryCore
    from xvla.train.implicit_projective_dag_artifact import export_implicit_projective_dag_artifact
    from xvla.train.implicit_projective_dag_mapped import open_mapped_implicit_projective_dag_artifact
    if type(node_count) is not int or not 10 <= node_count <= 100000:
        raise ValueError("bounded benchmark size differs")
    torch.set_num_threads(16)
    dtype = torch.float64
    leaves = max(2, (node_count + 2) // 3)
    level = [ImplicitNode(uid=index, label=f"benchmark.leaf{index}", core=UnaryCore(torch.eye(2, dtype=dtype), "physical"), physical_token=0) for index in range(leaves)]
    index = leaves
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
                continue
            core = ReducedQRBinaryCore(torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]], dtype=dtype), left_dimension=2, right_dimension=2, kind="benchmark_identity_rows")
            binary = ImplicitNode(uid=index, label=f"benchmark.{index}", core=core, children=(level[offset], level[offset + 1]))
            index += 1
            unary = ImplicitNode(uid=index, label=f"benchmark.{index}", core=UnaryCore(torch.eye(2, dtype=dtype), "benchmark_identity"), children=(binary,))
            index += 1
            next_level.append(unary)
        level = next_level
    root = level[0]
    node_count = index
    network = ImplicitProjectiveDAG(root=root, head=torch.eye(2, dtype=dtype), head_binary_exponent=0, token_count=1, feature_dimension=1,
        selected_token=0, mask=torch.ones(1, 1, dtype=dtype), claim_boundary="Synthetic bounded dispatch timing fixture only, not a scientific ODT certificate.",
        algorithm1_direct_rq_complete=True, algorithm1_scale_ledger_complete=True, algorithm1_certified_head_binary_exponent=0)
    destination = Path(tempfile.mkdtemp(prefix="odt_dispatch_")) / "artifact"
    receipt = export_implicit_projective_dag_artifact(network, destination)
    timings = []
    with open_mapped_implicit_projective_dag_artifact(destination, expected_manifest_sha256=receipt.manifest_sha256) as mapped:
        for batch in (1, 20):
            started = time.monotonic()
            output, execution = mapped.evaluate_boundary_quotient(torch.ones(batch, 1, 1, dtype=dtype), return_receipt=True)
            elapsed = time.monotonic() - started
            if not torch.equal(output, torch.ones(batch, 1, dtype=dtype)) or execution.node_evaluations != node_count:
                raise RuntimeError("dispatch fixture reconstruction failed")
            timings.append({"batch": batch, "nodes": node_count, "seconds": elapsed, "nodes_per_second": node_count / elapsed,
                "extrapolated_3964463node_seconds_not_a_production_benchmark": elapsed * 3964463 / node_count,
                "peak_live_values": execution.peak_live_values})
    result = {"passed": True, "fixture": "balanced unary2 and binary2 tree", "timings": timings,
        "numerical_guards": guards, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    import json
    print(json.dumps(result, sort_keys=True), flush=True)
    return result
