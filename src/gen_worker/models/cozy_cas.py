from __future__ import annotations

import asyncio
import errno
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import requests
from blake3 import blake3

from ..capability import InsufficientDiskError
from .errors import UrlExpiredError

_log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024

# Retry policy: transient network errors get a long budget; verification
# failures (size/blake3 mismatch) mean the source blob is likely corrupt —
# 2 re-downloads, then give up. UrlExpiredError / ENOSPC never retry.
_TRANSIENT_MAX_TRIES = 30
_MAX_RETRY_TIME_S = 3600.0
_VERIFY_MAX_FAILURES = 3  # initial try + 2 retries
_RETRY_BACKOFF_CAP_S = 30.0


def _norm_rel_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    s = s.lstrip("/")
    if not s or s == ".":
        raise ValueError("empty relative path")
    if ".." in s.split("/"):
        raise ValueError("path traversal not allowed")
    return s


def _check_status(resp: requests.Response) -> None:
    """4xx (except 408/429) on a presigned URL is dead — no retry can help."""
    sc = int(resp.status_code)
    if 400 <= sc < 500 and sc not in (408, 429):
        raise UrlExpiredError(f"download URL rejected with HTTP {sc}", status_code=sc)
    resp.raise_for_status()


async def _download_one_file(
    url: str,
    dst: Path,
    expected_size: int,
    expected_blake3: str,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> None:
    loop = asyncio.get_running_loop()
    started = time.monotonic()
    transient_failures = 0
    verify_failures = 0
    while True:
        try:
            await loop.run_in_executor(
                None,
                lambda: _download_one_file_sync(
                    url, dst, expected_size, expected_blake3, on_bytes=on_bytes
                ),
            )
            return
        except (UrlExpiredError, InsufficientDiskError):
            raise
        except (requests.RequestException, ValueError, OSError) as exc:
            if getattr(exc, "errno", None) == errno.ENOSPC:
                raise InsufficientDiskError(
                    f"insufficient disk while downloading {dst.name}", path=str(dst.parent)
                ) from exc
            if isinstance(exc, ValueError):
                verify_failures += 1
                failures, cap = verify_failures, _VERIFY_MAX_FAILURES
            else:
                transient_failures += 1
                failures, cap = transient_failures, _TRANSIENT_MAX_TRIES
            if failures >= cap or time.monotonic() - started >= _MAX_RETRY_TIME_S:
                raise
            delay = random.uniform(0.5, 1.0) * min(_RETRY_BACKOFF_CAP_S, 2.0 ** failures)
            _log.warning("download of %s failed (%s, failure %d/%d): %s; retrying in %.1fs",
                         dst.name, type(exc).__name__, failures, cap, exc, delay)
            await asyncio.sleep(delay)


def _download_one_file_sync(
    url: str,
    dst: Path,
    expected_size: int,
    expected_blake3: str,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> None:
    """Download a single file with HTTP Range resume, size + blake3 validation.

    Caller runs this in a worker thread; the transfer itself uses the same
    requests stack as the Tensorhub control plane fallback path.
    """
    import logging

    log = logging.getLogger("gen_worker.download")

    def _human_size(n: int) -> str:
        if n >= 1 << 30:
            return f"{n / (1 << 30):.1f}GB"
        if n >= 1 << 20:
            return f"{n / (1 << 20):.1f}MB"
        if n >= 1 << 10:
            return f"{n / (1 << 10):.1f}KB"
        return f"{n}B"

    # Already downloaded and valid?
    if dst.exists():
        try:
            if expected_size and dst.stat().st_size != expected_size:
                raise ValueError("size mismatch")
            if expected_blake3:
                got = _blake3_file(dst)
                if got.lower() != expected_blake3.lower():
                    raise ValueError("blake3 mismatch")
            log.info("download_cached path=%s size=%s", dst.name, _human_size(dst.stat().st_size))
            return
        except Exception:
            pass  # re-download

    tmp = dst.with_suffix(dst.suffix + ".part")

    # Resume from partial download if available.
    offset = 0
    if tmp.exists():
        try:
            offset = tmp.stat().st_size
        except OSError:
            offset = 0
        if expected_size and offset > expected_size:
            tmp.unlink(missing_ok=True)
            offset = 0

    # Partial file already complete? Validate and finalize.
    if offset and expected_size and offset == expected_size:
        got = _blake3_file(tmp)
        if expected_blake3 and got.lower() != expected_blake3.lower():
            log.warning("partial_corrupt path=%s (blake3 mismatch, restarting)", dst.name)
            tmp.unlink(missing_ok=True)
            offset = 0
        else:
            tmp.rename(dst)
            log.info("download_resumed_complete path=%s size=%s", dst.name, _human_size(expected_size))
            return

    headers: Dict[str, str] = {}
    mode = "wb"
    if offset and expected_size:
        headers["Range"] = f"bytes={offset}-"
        mode = "ab"
        log.info("download_resume path=%s offset=%s/%s (%s/%s)",
                 dst.name, offset, expected_size,
                 _human_size(offset), _human_size(expected_size))
    else:
        log.info("download_start path=%s size=%s blake3=%s",
                 dst.name, _human_size(expected_size) if expected_size else "unknown",
                 (expected_blake3 or "n/a")[:16])

    def _stream(resp: requests.Response, *, write_mode: str, start: int) -> None:
        downloaded = start
        last_log = start
        log_every = max(expected_size // 10, 50 << 20) if expected_size else (100 << 20)
        with open(tmp, write_mode) as f:
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
                if expected_size and downloaded > expected_size:
                    raise ValueError(f"download exceeded expected size ({downloaded} > {expected_size})")
                if downloaded - last_log >= log_every:
                    pct = f" ({100 * downloaded // expected_size}%)" if expected_size else ""
                    log.info("download_progress path=%s downloaded=%s%s",
                             dst.name, _human_size(downloaded), pct)
                    last_log = downloaded

    with requests.Session() as session:
        with session.get(url, headers=headers, stream=True, timeout=(60, 180)) as resp:
            content_range = str(resp.headers.get("Content-Range") or "").strip()

            # Server ignored Range or returned unexpected range start?
            # Restart from byte 0 to avoid corrupted appends.
            if offset and (
                resp.status_code == 200
                or (resp.status_code == 206 and not content_range.startswith(f"bytes {offset}-"))
            ):
                log.info("download_range_ignored path=%s status=%s (restarting from 0)", dst.name, resp.status_code)
                resp.close()
                with session.get(url, stream=True, timeout=(60, 180)) as resp2:
                    _check_status(resp2)
                    _stream(resp2, write_mode="wb", start=0)
            else:
                _check_status(resp)
                _stream(resp, write_mode=mode, start=offset)

    # Validate.
    actual_size = tmp.stat().st_size
    if expected_size and actual_size != expected_size:
        log.error("download_size_mismatch path=%s expected=%s got=%s", dst.name, expected_size, actual_size)
        tmp.unlink(missing_ok=True)
        raise ValueError(f"size mismatch (expected {expected_size}, got {actual_size})")

    if expected_blake3:
        got = _blake3_file(tmp)
        if got.lower() != expected_blake3.lower():
            log.error("download_blake3_mismatch path=%s expected=%s got=%s",
                      dst.name, expected_blake3[:16], got[:16])
            tmp.unlink(missing_ok=True)
            raise ValueError("blake3 mismatch")

    # Atomic finalize.
    tmp.replace(dst)
    log.info("download_done path=%s size=%s", dst.name, _human_size(actual_size))


def _blake3_file(path: Path, chunk_size: int = _DOWNLOAD_CHUNK_BYTES) -> str:
    # Fan BLAKE3 across cores via AUTO threading (issue #269). Used on
    # download-verify path; same throughput win as the upload-side hash.
    h = blake3(max_threads=blake3.AUTO)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
