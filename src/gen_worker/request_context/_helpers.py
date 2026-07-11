"""Module-level helpers for RequestContext: parsing, SSRF checks, hashing, etc.

These are pure utilities with no dependency on RequestContext or
_RequestOutputStream. Pulled out of the monolithic request_context.py so the
class files only hold class bodies.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import re
import socket
import urllib.parse
from typing import Any, Dict

from ..api.errors import OutputTooLargeError

logger = logging.getLogger(__name__)


_PUBLIC_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_STALE_MIRROR_CLAIM_ERROR_CODES = {"source_version_not_found", "source_variants_not_found"}
_MAX_OUTPUT_FILE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB hard cap per file.
_FILE_API_HTTP_TIMEOUT_S = 60
_FILE_API_STREAM_CHUNK_TIMEOUT_S = 120
_FILE_API_STREAM_FINALIZE_TIMEOUT_S = 600
_FILE_API_STREAM_REPLAY_TIMEOUT_S = 600
_FILE_API_STREAM_ABORT_TIMEOUT_S = 15


def _infer_mime_type(ref: str, head: bytes) -> str:
    # Prefer magic bytes when available.
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    # Fall back to extension.
    import mimetypes

    guessed, _ = mimetypes.guess_type(ref)
    return guessed or "application/octet-stream"


def _normalize_output_ref(ref: str) -> str:
    out = str(ref or "").strip()
    if not out:
        raise ValueError("invalid ref")
    lower = out.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        raise ValueError("output ref must be a logical file ref, not a URL")
    return out.lstrip("/")


def _infer_tensors_format(ref_or_path: str) -> str:
    leaf = str(ref_or_path or "").strip().lower()
    if leaf.endswith(".safetensors"):
        return "safetensors"
    if leaf.endswith(".bin"):
        return "bin"
    if leaf.endswith(".pt"):
        return "pt"
    if leaf.endswith(".pth"):
        return "pth"
    if leaf.endswith(".ckpt"):
        return "ckpt"
    return "unknown"


def _require_worker_capability_token() -> str:
    # The per-request worker_capability_token is plumbed from
    # JobExecutionRequest.worker_capability_token via RequestContext.
    # Callers should prefer RequestContext._get_worker_capability_token(),
    # which uses the per-request token directly and only falls back here
    # when none is available. Without a per-request token, file
    # operations cannot proceed.
    raise RuntimeError("worker_capability_token is required for file operations")


def _parse_owner_repo(value: str) -> tuple[str, str]:
    raw = str(value or "").strip().strip("/")
    if "/" not in raw:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    owner, repo = raw.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    return owner, repo


def _decode_unverified_jwt_claims(token: str) -> Dict[str, Any]:
    raw = str(token or "").strip()
    if raw.count(".") < 2:
        return {}
    try:
        parts = raw.split(".")
        payload_b64 = parts[1]
        pad = "=" * ((4 - (len(payload_b64) % 4)) % 4)
        payload = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _enforce_output_file_size_limit(size_bytes: int) -> None:
    size = int(size_bytes)
    if size < 0:
        raise ValueError("size_bytes must be non-negative")
    if size > _MAX_OUTPUT_FILE_BYTES:
        raise OutputTooLargeError(size_bytes=size, max_bytes=_MAX_OUTPUT_FILE_BYTES)


def _is_private_ip_str(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _url_is_blocked(url_str: str) -> bool:
    try:
        u = urllib.parse.urlparse(url_str)
    except Exception:
        return True
    if u.scheme not in ("http", "https"):
        return True
    host = (u.hostname or "").strip()
    if not host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = str(sockaddr[0])
        if _is_private_ip_str(ip_str):
            return True
    return False


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
