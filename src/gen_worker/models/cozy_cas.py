from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

import aiohttp
import backoff
from blake3 import blake3

_log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024


def _norm_rel_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    s = s.lstrip("/")
    if not s or s == ".":
        raise ValueError("empty relative path")
    if ".." in s.split("/"):
        raise ValueError("path traversal not allowed")
    return s


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError),
    max_tries=30,
    max_time=3600,
    factor=1,
    max_value=30,  # cap backoff at 30s between retries
)
async def _download_one_file(url: str, dst: Path, expected_size: int, expected_blake3: str) -> None:
    """Download a single file with HTTP Range resume, size + blake3 validation.

    Fully async — no blocking calls that would stall the event loop.
    Caller is responsible for ensuring only one coroutine downloads a given
    dst at a time (dedup by digest in _ensure_blobs).
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

    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=float(os.getenv("WORKER_MODEL_DOWNLOAD_SOCK_CONNECT_TIMEOUT_S", "60")),
        sock_read=float(os.getenv("WORKER_MODEL_DOWNLOAD_SOCK_READ_TIMEOUT_S", "180")),
    )
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

    async def _stream(resp: aiohttp.ClientResponse, *, write_mode: str, start: int) -> None:
        downloaded = start
        last_log = start
        log_every = max(expected_size // 10, 50 << 20) if expected_size else (100 << 20)
        with open(tmp, write_mode) as f:
            async for chunk in resp.content.iter_chunked(_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if expected_size and downloaded > expected_size:
                    raise ValueError(f"download exceeded expected size ({downloaded} > {expected_size})")
                if downloaded - last_log >= log_every:
                    pct = f" ({100 * downloaded // expected_size}%)" if expected_size else ""
                    log.info("download_progress path=%s downloaded=%s%s",
                             dst.name, _human_size(downloaded), pct)
                    last_log = downloaded

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            content_range = str(resp.headers.get("Content-Range") or "").strip()

            # Server ignored Range or returned unexpected range start?
            # Restart from byte 0 to avoid corrupted appends.
            if offset and (
                resp.status == 200
                or (resp.status == 206 and not content_range.startswith(f"bytes {offset}-"))
            ):
                log.info("download_range_ignored path=%s status=%s (restarting from 0)", dst.name, resp.status)
                resp.release()
                async with session.get(url) as resp2:
                    resp2.raise_for_status()
                    await _stream(resp2, write_mode="wb", start=0)
            else:
                resp.raise_for_status()
                await _stream(resp, write_mode=mode, start=offset)

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
    h = blake3()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
