"""Materialize URL-ref input Assets before the handler runs.

At invoke time tensorhub validates a typed media field's remote URL
(``normalizeTypedRemoteMediaInputs``) and rewrites it into an Asset whose
``ref`` IS the approved http(s) URL, carrying the validation caps
(``url_max_bytes``, ``url_allowed_mime_types``). The worker downloads those
refs into a per-request directory and sets ``local_path`` so tenant code
(``payload.image.local_path`` / ``gen_worker.io.read_image``) just works.

Non-URL refs are left untouched.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import msgspec

from .api.errors import ValidationError
from .api.types import Asset
from .request_context._helpers import _infer_mime_type, _url_is_blocked

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 50 << 20  # matches tensorhub's default media cap
_DOWNLOAD_TIMEOUT_S = 120
_MAX_WALK_DEPTH = 8
_CHUNK = 1 << 20

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}


def inputs_dir_for_request(request_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "gen-worker-inputs" / str(request_id)


def _iter_assets(obj: Any, depth: int = 0) -> Iterator[Asset]:
    if depth > _MAX_WALK_DEPTH:
        return
    if isinstance(obj, Asset):
        yield obj
        return
    if isinstance(obj, msgspec.Struct):
        for name in obj.__struct_fields__:
            yield from _iter_assets(getattr(obj, name), depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_assets(item, depth + 1)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_assets(item, depth + 1)


def _is_http_url(ref: str) -> bool:
    low = str(ref or "").strip().lower()
    return low.startswith("http://") or low.startswith("https://")


def _mime_allowed(mime: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    mime = (mime or "").lower()
    for a in allowed:
        a = str(a or "").lower()
        if not a:
            continue
        if mime == a:
            return True
        if a.endswith("/*") and mime.startswith(a[:-1]):
            return True
    return False


def _download(asset: Asset, dest_dir: Path, index: int) -> None:
    url = str(asset.ref).strip()
    if _url_is_blocked(url):
        raise ValidationError(f"input asset URL is not allowed: {url!r}")
    cap = int(asset.url_max_bytes or 0) or _DEFAULT_MAX_BYTES
    req = urllib.request.Request(url, headers={"User-Agent": "gen-worker"})
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f"input-{index}.part"
    total = 0
    head = b""
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            if not head:
                head = chunk[:64]
            total += len(chunk)
            if total > cap:
                tmp.unlink(missing_ok=True)
                raise ValidationError(
                    f"input asset exceeds size cap ({cap} bytes): {url!r}")
            out.write(chunk)
    mime = _infer_mime_type(url, head)
    if not _mime_allowed(mime, tuple(asset.url_allowed_mime_types or ())):
        tmp.unlink(missing_ok=True)
        raise ValidationError(
            f"input asset content type {mime!r} not allowed for {url!r}")
    final = dest_dir / f"input-{index}{_EXT_BY_MIME.get(mime, '')}"
    tmp.rename(final)
    asset.local_path = str(final)
    asset.size_bytes = total
    if not asset.mime_type:
        asset.mime_type = mime
    logger.info("materialized input asset %s -> %s (%d bytes, %s)",
                url.split("?")[0], final, total, mime)


def materialize_input_assets(payload: Any, request_id: str) -> int:
    """Download every URL-ref Asset in ``payload`` into the request's input
    dir and set ``local_path``. Returns the number materialized. Blocking —
    call via ``asyncio.to_thread``."""
    count = 0
    dest = inputs_dir_for_request(request_id)
    for asset in _iter_assets(payload):
        if asset.local_path or not _is_http_url(asset.ref):
            continue
        _download(asset, dest, count)
        count += 1
    return count


def cleanup_input_assets(request_id: str) -> None:
    """Best-effort removal of the request's materialized input files."""
    rid = str(request_id or "").strip()
    if not rid or "/" in rid or rid in (".", ".."):
        return
    shutil.rmtree(inputs_dir_for_request(rid), ignore_errors=True)


__all__ = [
    "cleanup_input_assets",
    "inputs_dir_for_request",
    "materialize_input_assets",
]
