"""Upload one authenticated Athena object using short-lived object-scoped URLs.

No account credential is accepted.  The JSON request is read from stdin and is
never logged.  Responses contain hashes, sizes, and status only.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import stat
import sys
import urllib.parse


def _snapshot(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_nlink)


@contextlib.contextmanager
def open_source(path: str, *, allowed_roots=(), allowed_files=()):
    """Hold one immutable inode through authentication and every upload part."""
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("input must be an explicit absolute file")
    if str(candidate) not in allowed_files and not any(
            Path(root).is_absolute() and Path(root) in candidate.parents for root in allowed_roots):
        raise ValueError("input is outside the immutable campaign scope")
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    try:
        for component in candidate.parts[1:-1]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = following
        descriptor = os.open(candidate.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222:
            raise ValueError("input must be a read-only physical single-link file")
        yield descriptor, before
        if (_snapshot(os.fstat(descriptor)) != _snapshot(before)
                or _snapshot(os.stat(candidate.name, dir_fd=parent, follow_symlinks=False)) != _snapshot(before)):
            raise RuntimeError("authenticated source inode changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def descriptor_metadata(path: str, descriptor: int, before) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(8 << 20, before.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("authenticated source ended early")
        sha.update(chunk)
        md5.update(chunk)
        offset += len(chunk)
    if _snapshot(os.fstat(descriptor)) != _snapshot(before):
        raise RuntimeError("input changed during authentication")
    return {"path": path, "size": before.st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def metadata(path: str, *, allowed_roots=(), allowed_files=()) -> dict:
    with open_source(path, allowed_roots=allowed_roots, allowed_files=allowed_files) as (descriptor, before):
        return descriptor_metadata(path, descriptor, before)


def request(method: str, url: str, *, data=None, headers=None) -> tuple[dict, bytes]:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise ValueError("upload destination is not a secure scoped URL")
    if not (host.endswith(".amazonaws.com") or host.endswith(".r2.cloudflarestorage.com") or host.endswith(".modal.com") or host.endswith(".modalusercontent.com")):
        raise ValueError("upload URL is outside the Modal object-storage providers")
    connection = http.client.HTTPSConnection(host, timeout=300, blocksize=1024 * 1024)
    try:
        connection.request(method, urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, "")), body=data, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"object upload HTTP status {response.status}")
        return dict(response.getheaders()), body
    finally:
        connection.close()


class Segment:
    def __init__(self, descriptor: int, offset: int, length: int):
        self.descriptor = descriptor
        self.offset = offset
        self.remaining = length
        self.md5 = hashlib.md5()

    def read(self, size: int = -1) -> bytes:
        size = min(self.remaining, size if size >= 0 else self.remaining)
        value = os.pread(self.descriptor, size, self.offset)
        if size and not value:
            raise RuntimeError("authenticated source ended during upload")
        self.offset += len(value)
        self.remaining -= len(value)
        self.md5.update(value)
        return value

    def close(self):
        pass  # The caller owns the stable inode, including all parallel parts.


def upload_part(descriptor: int, offset: int, length: int, url: str, content_md5: str | None = None) -> str:
    segment = Segment(descriptor, offset, length)
    headers = {"Content-Length": str(length)}
    if content_md5:
        headers["Content-MD5"] = content_md5
        headers["Content-Type"] = "application/octet-stream"
    try:
        response, _body = request("PUT", url, data=segment, headers=headers)
        remote = next((value for key, value in response.items() if key.lower() == "etag"), "").strip().removeprefix("W/").strip('"')
        if segment.remaining or remote != segment.md5.hexdigest():
            raise RuntimeError("object part checksum differs")
        return remote
    finally:
        segment.close()


def perform(value: dict) -> dict:
    scope = {"allowed_roots": value.get("allowed_roots", ()), "allowed_files": value.get("allowed_files", ())}
    with open_source(value["path"], **scope) as (descriptor, before):
        checked = descriptor_metadata(value["path"], descriptor, before)
        result = _perform_open(value, checked, descriptor)
    return result


def _perform_open(value: dict, checked: dict, descriptor: int) -> dict:
    if value.get("operation") == "inspect":
        return checked
    if value.get("operation") != "upload" or checked["sha256"] != value["sha256"] or checked["size"] != value["size"]:
        raise RuntimeError("upload identity differs")
    if value["type"] == "single":
        upload_part(descriptor, 0, checked["size"], value["url"], value["content_md5"])
    elif value["type"] == "multipart":
        width = value["part_length"]
        urls = value["urls"]
        if width <= 0 or len(urls) != (checked["size"] + width - 1) // width:
            raise ValueError("multipart inventory differs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(upload_part, descriptor, index * width, min(width, checked["size"] - index * width), url) for index, url in enumerate(urls)]
            etags = [future.result() for future in futures]
        expected = hashlib.md5(b"".join(bytes.fromhex(etag) for etag in etags)).hexdigest() + f"-{len(etags)}"
        xml = "<CompleteMultipartUpload>" + "".join(f'<Part><PartNumber>{index}</PartNumber><ETag>"{etag}"</ETag></Part>' for index, etag in enumerate(etags, 1)) + "</CompleteMultipartUpload>"
        encoded = xml.encode("ascii")
        _headers, body = request("POST", value["completion_url"], data=encoded, headers={"Content-Length": str(len(encoded))})
        if expected.encode() not in body:
            raise RuntimeError("assembled object checksum differs")
    else:
        raise ValueError("unsupported upload descriptor")
    return {**checked, "uploaded": True}


def main() -> None:
    print(json.dumps(perform(json.load(sys.stdin)), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print exception URLs, payloads, or object-scoped credentials.
        print(json.dumps({"uploaded": False, "error_type": type(error).__name__, "error": "remote authenticated upload failed"}), flush=True)
        raise SystemExit(1)
