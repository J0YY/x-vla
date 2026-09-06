"""Bounded Modal CPU-memory admission probe for the canonical ODT campaign.

This probe imports no scientific code, runs for at most 60 seconds per admitted
container, and allocates no large tensors.  It records actual service admission
and cgroup limits rather than treating host RAM as a resource guarantee.
"""

from __future__ import annotations

import argparse
import json
import time

import modal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, nargs="+", default=[900, 700, 500, 256])
    parser.add_argument("--gpu", default=None)
    options = parser.parse_args()
    application = modal.App.lookup("xvla-odt-dimension-curve-resource-probe", create_if_missing=True)
    image = modal.Image.from_registry("python:3.10.19-slim-bookworm")
    code = """import json,os,pathlib
paths=['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.high','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']
print(json.dumps({'pid':os.getpid(),'limits':{p:pathlib.Path(p).read_text().strip() for p in paths if pathlib.Path(p).exists()}},sort_keys=True))
"""
    for gib in options.gib:
        started = time.monotonic()
        sandbox = None
        result = {"requested_gib": gib, "requested_physical_cpus": 16, "gpu": options.gpu, "admitted": False}
        try:
            sandbox = modal.Sandbox.create(
                "python", "-c", code, app=application, image=image,
                cpu=(16.0, 16.0), memory=(gib * 1024, gib * 1024), timeout=60,
                gpu=options.gpu,
            )
            result["sandbox_id"] = sandbox.object_id
            sandbox.wait()
            result["returncode"] = sandbox.returncode
            result["stdout"] = sandbox.stdout.read()
            result["stderr"] = sandbox.stderr.read()
            result["admitted"] = sandbox.returncode == 0
        except Exception as error:
            result["error_type"] = type(error).__name__
            result["error"] = str(error)
        finally:
            if sandbox is not None:
                sandbox.terminate()
        result["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["admitted"]:
            break


if __name__ == "__main__":
    main()
