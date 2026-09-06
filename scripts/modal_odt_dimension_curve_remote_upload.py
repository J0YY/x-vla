"""Upload one authenticated Athena object using short-lived object-scoped URLs.

No account credential is accepted.  The JSON request is read from stdin and is
never logged.  Responses contain hashes, sizes, and status only.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import stat
import sys
import urllib.parse


def metadata(path: str) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts or not str(candidate).startswith("/work/joy/"):
        raise ValueError("input must be an explicit authenticated Athena work file")
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("input must be a physical single-link file")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    after = os.lstat(candidate)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("input changed during authentication")
    return {"path": path, "size": before.st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


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
    def __init__(self, path: str, offset: int, length: int):
        self.stream = open(path, "rb")
        self.stream.seek(offset)
        self.remaining = length
        self.md5 = hashlib.md5()

    def read(self, size: int = -1) -> bytes:
        size = min(self.remaining, size if size >= 0 else self.remaining)
        value = self.stream.read(size)
        self.remaining -= len(value)
        self.md5.update(value)
        return value

    def close(self):
        self.stream.close()


def upload_part(path: str, offset: int, length: int, url: str, content_md5: str | None = None) -> str:
    segment = Segment(path, offset, length)
    headers = {"Content-Length": str(length)}
    if content_md5:
        headers["Content-MD5"] = content_md5
        headers["Content-Type"] = "application/octet-stream"
    try:
        response, _body = request("PUT", url, data=segment, headers=headers)
        remote = next((value for key, value in response.items() if key.lower() == "etag"), "").strip('"')
        if segment.remaining or remote != segment.md5.hexdigest():
            raise RuntimeError("object part checksum differs")
        return remote
    finally:
        segment.close()


def main() -> None:
    value = json.load(sys.stdin)
    checked = metadata(value["path"])
    if value.get("operation") == "inspect":
        print(json.dumps(checked, sort_keys=True))
        return
    if value.get("operation") != "upload" or checked["sha256"] != value["sha256"] or checked["size"] != value["size"]:
        raise RuntimeError("upload identity differs")
    if value["type"] == "single":
        upload_part(checked["path"], 0, checked["size"], value["url"], value["content_md5"])
    elif value["type"] == "multipart":
        width = value["part_length"]
        urls = value["urls"]
        if width <= 0 or len(urls) != (checked["size"] + width - 1) // width:
            raise ValueError("multipart inventory differs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(upload_part, checked["path"], index * width, min(width, checked["size"] - index * width), url) for index, url in enumerate(urls)]
            etags = [future.result() for future in futures]
        expected = hashlib.md5(b"".join(bytes.fromhex(etag) for etag in etags)).hexdigest() + f"-{len(etags)}"
        xml = "<CompleteMultipartUpload>" + "".join(f'<Part><PartNumber>{index}</PartNumber><ETag>"{etag}"</ETag></Part>' for index, etag in enumerate(etags, 1)) + "</CompleteMultipartUpload>"
        encoded = xml.encode("ascii")
        _headers, body = request("POST", value["completion_url"], data=encoded, headers={"Content-Length": str(len(encoded))})
        if expected.encode() not in body:
            raise RuntimeError("assembled object checksum differs")
    else:
        raise ValueError("unsupported upload descriptor")
    if metadata(value["path"]) != checked:
        raise RuntimeError("input changed after upload")
    print(json.dumps({**checked, "uploaded": True}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print exception URLs, payloads, or object-scoped credentials.
        print(json.dumps({"uploaded": False, "error_type": type(error).__name__, "error": "remote authenticated upload failed"}), flush=True)
        raise SystemExit(1)
