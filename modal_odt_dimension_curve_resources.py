"""Bounded Modal CPU-memory admission probe for the canonical ODT campaign.

This probe imports no scientific code, runs for at most 60 seconds per admitted
container, and allocates no large tensors.  It records actual service admission
and cgroup limits rather than treating host RAM as a resource guarantee.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import modal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, nargs="+", default=[900, 700, 500, 256])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--concurrent", type=int, default=1)
    options = parser.parse_args()
    if not 1 <= options.concurrent <= 6:
        raise ValueError("admission probe is bounded to one through six reservations")
    application = modal.App.lookup("xvla-odt-dimension-curve-resource-probe", create_if_missing=True)
    image = modal.Image.from_registry("python:3.10.19-slim-bookworm")
    if options.concurrent > 1:
        if options.gib != [336] or options.gpu != "L4":
            raise ValueError("concurrent probe is fixed to the requested336GiB/16CPU/L4 workers")
        concurrent_code = "import json,pathlib,time; start=time.time(); paths=['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.high','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']; limits={p:pathlib.Path(p).read_text().strip() for p in paths if pathlib.Path(p).exists()}; time.sleep(45); print(json.dumps({'start':start,'end':time.time(),'limits':limits}),flush=True)"
        def reservation(index):
            sandbox = None
            try:
                sandbox = modal.Sandbox.create("python", "-c", concurrent_code, app=application, image=image,
                    cpu=(16., 16.), memory=(344064, 344064), timeout=60, gpu="L4")
                sandbox.wait()
                if sandbox.returncode != 0:
                    return {"index": index, "sandbox_id": sandbox.object_id, "admitted": False, "stderr": sandbox.stderr.read()}
                return {"index": index, "sandbox_id": sandbox.object_id, "admitted": True, **json.loads(sandbox.stdout.read())}
            except Exception as error:
                return {"index": index, "admitted": False, "error": str(error), "error_type": type(error).__name__}
            finally:
                if sandbox is not None:
                    sandbox.terminate()
        with ThreadPoolExecutor(max_workers=options.concurrent) as pool:
            receipts = list(pool.map(reservation, range(options.concurrent)))
        overlap = min(row["end"] for row in receipts) - max(row["start"] for row in receipts) if all(row["admitted"] for row in receipts) else 0.
        print(json.dumps({"reservations": receipts, "six_simultaneously_admitted": len(receipts) == 6 and overlap > 0,
            "all_reservations_overlap_seconds": max(0., overlap), "cleanup": "all_created_sandboxes_terminated"}, sort_keys=True), flush=True)
        return
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
