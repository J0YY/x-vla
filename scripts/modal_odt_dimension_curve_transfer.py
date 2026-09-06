"""Direct Athena-to-Modal transfer without exporting account credentials.

The installed Modal 1.3.5 SDK's BlobCreate/MountPutFile/VolumePutFiles protocol is
used explicitly.  Only short-lived object upload URLs pass over authenticated
SSH stdin.  All files remain content-addressed, and overwrite is prohibited.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time

from modal.client import _Client
from modal_proto import api_pb2


REMOTE_HELPER = "/work/joy/modal_odt_dimension_curve_remote_upload.py"
REMOTE_HELPER_PYTHON = "/users/joy/miniconda3/envs/safesae-openvla/bin/python"


def remote(request: dict) -> dict:
    completed = subprocess.run(["ssh", "athena", REMOTE_HELPER_PYTHON, REMOTE_HELPER],
        input=json.dumps(request), capture_output=True, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError("Athena object transfer failed; URLs and credentials redacted")
    result = json.loads(completed.stdout)
    if type(result) is not dict:
        raise RuntimeError("Athena transfer response is malformed")
    return result


async def transfer(path: str, expected_sha256: str, destination: str) -> dict:
    started = time.monotonic()
    if not destination.startswith("/odt_dimension_curve_v1/") or ".." in destination.split("/"):
        raise ValueError("destination must be in this campaign's Modal subtree")
    metadata = await asyncio.to_thread(remote, {"operation": "inspect", "path": path})
    if metadata["sha256"] != expected_sha256:
        raise RuntimeError("Athena file differs from trusted manifest")
    client = await _Client.from_env()
    volume = await client.stub.VolumeGetOrCreate(api_pb2.VolumeGetOrCreateRequest(deployment_name="xvla-data", environment_name="main"))
    if volume.metadata.version != 1:
        raise RuntimeError("this validated uploader requires VolumeFS v1")
    existing = await client.stub.MountPutFile(api_pb2.MountPutFileRequest(sha256_hex=expected_sha256))
    uploaded = False
    if not existing.exists:
        md5 = base64.b64encode(bytes.fromhex(metadata["md5"])).decode()
        created = await client.stub.BlobCreate(api_pb2.BlobCreateRequest(content_md5=md5,
            content_sha256_base64=base64.b64encode(bytes.fromhex(expected_sha256)).decode(), content_length=metadata["size"]))
        request = {**metadata, "operation": "upload"}
        if created.WhichOneof("upload_types_oneof") == "multiparts":
            part = created.multiparts.items[0]
            request.update(type="multipart", part_length=part.part_length, urls=list(part.upload_urls), completion_url=part.completion_url)
        else:
            request.update(type="single", url=created.upload_urls.items[0], content_md5=md5)
        report = await asyncio.to_thread(remote, request)
        if report.get("uploaded") is not True or report["sha256"] != expected_sha256 or report["size"] != metadata["size"]:
            raise RuntimeError("remote upload receipt differs")
        attached = await client.stub.MountPutFile(api_pb2.MountPutFileRequest(data_blob_id=created.blob_ids[0], sha256_hex=expected_sha256))
        if not attached.exists:
            raise RuntimeError("uploaded content did not become available")
        uploaded = True
    await client.stub.VolumePutFiles(api_pb2.VolumePutFilesRequest(volume_id=volume.volume_id,
        files=[api_pb2.MountFile(filename=destination, sha256_hex=expected_sha256, mode=0o444)],
        disallow_overwrite_existing_files=True))
    return {"source": path, "destination": destination, "sha256": expected_sha256,
        "bytes": metadata["size"], "uploaded_directly_from_athena": uploaded,
        "elapsed_seconds": time.monotonic() - started, "permanent_credentials_transferred": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", required=True)
    options = parser.parse_args()
    print(json.dumps(asyncio.run(transfer(options.source, options.sha256, options.destination)), sort_keys=True))


if __name__ == "__main__":
    main()
