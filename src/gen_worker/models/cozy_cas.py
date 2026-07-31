from __future__ import annotations

import asyncio
import errno
import logging
import os
import random
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from ..capability import InsufficientDiskError
from ..stall import ProgressFloor
from .chunk_cas import DigestMismatch, parse_cas_ref, verify_file_digest
from .errors import UrlExpiredError

_log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024

# Retry policy. Verification failures (size/digest mismatch) mean the source
# blob is likely corrupt — 2 re-downloads, then give up. UrlExpiredError /
# ENOSPC never retry.
#
# gw#666 (th#1166 finding C): the transient half used to give up after
# `_MAX_RETRY_TIME_S = 3600.0` of WALL time, however many bytes had landed —
# a clock cannot tell a 200GB blob crawling in over a bad pod uplink from a
# wedge, and resume-on-retry means the slow case was genuinely converging.
# The give-up is now a PROGRESS question: `_TRANSIENT_MAX_TRIES` counts
# CONSECUTIVE transient failures that failed to move the byte floor, and any
# attempt that delivers `_RETRY_PROGRESS_FLOOR_BYTES` resets the count. A
# link that keeps landing bytes is never "30 strikes and out"; a link that
# delivers nothing gives up in 30 attempts as before. The floor matches
# models/download.py's gw#655 progress-rate floor, and the per-attempt
# network timeouts (60s connect / 180s read) still bound each try.
_TRANSIENT_MAX_TRIES = 30
_RETRY_PROGRESS_FLOOR_BYTES = 8 * 1024 * 1024
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
    expected_digest: str,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> None:
    """th#1303 S1: ``expected_digest`` is ALGORITHM-TAGGED and MANDATORY.

    It used to be a bare blake3 hex guarded by ``if expected_blake3:`` — the
    vacuous-guard shape: a v2 entry carries no blake3, so the guard was false
    and the file was published unverified. There is no unverified path through
    this function any more; an empty or untagged digest raises before a single
    byte is fetched.
    """
    # Validate the ref ONCE, before the retry loop: an absent or unhashable
    # digest is a permanent refusal, and retrying it three times only delays
    # the same answer while looking like a flaky network.
    if not str(expected_digest or "").strip():
        raise ValueError(f"download of {dst.name}: no expected digest — refusing")
    hash_algo, _ = parse_cas_ref(expected_digest)
    if hash_algo != "sha256":
        raise ValueError(
            f"download of {dst.name}: digest algorithm {hash_algo!r} is no longer "
            "verifiable (th#1303 S1) — refusing rather than fetching unchecked"
        )
    loop = asyncio.get_running_loop()
    transient_failures = 0
    verify_failures = 0
    # Bytes this call has actually pulled down, across attempts — the only
    # honest evidence that the transfer is still alive.
    delivered = 0
    floor = ProgressFloor(_RETRY_PROGRESS_FLOOR_BYTES)

    def _count_bytes(n: int) -> None:
        nonlocal delivered
        delivered += n
        if on_bytes is not None:
            on_bytes(n)

    # Writer-unique for the lifetime of this call (stable across its own
    # retries, so HTTP Range resume-on-retry still works) but distinct from
    # every OTHER writer of this digest — including a different PROCESS.
    # dst's CAS root may now be a network volume shared by concurrent
    # same-endpoint pods; a fixed ".part" name would let two pods racing on
    # the same missing blob interleave writes into the same file (th#850).
    writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    while True:
        try:
            await loop.run_in_executor(
                None,
                lambda: _download_one_file_sync(
                    url, dst, expected_size, expected_digest,
                    on_bytes=_count_bytes, writer_id=writer_id,
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
                # Bytes that landed since the last give-up check are real
                # progress; a flaky link that keeps delivering starts its
                # strike count over rather than accumulating toward a cap.
                if floor.cleared(delivered):
                    transient_failures = 0
                transient_failures += 1
                failures, cap = transient_failures, _TRANSIENT_MAX_TRIES
            if failures >= cap:
                raise
            delay = random.uniform(0.5, 1.0) * min(_RETRY_BACKOFF_CAP_S, 2.0 ** failures)
            _log.warning("download of %s failed (%s, failure %d/%d): %s; retrying in %.1fs",
                         dst.name, type(exc).__name__, failures, cap, exc, delay)
            await asyncio.sleep(delay)


def _download_one_file_sync(
    url: str,
    dst: Path,
    expected_size: int,
    expected_digest: str,
    on_bytes: Optional[Callable[[int], None]] = None,
    writer_id: Optional[str] = None,
) -> None:
    """Download a single file with HTTP Range resume, size + digest validation.

    The digest is algorithm-tagged and the hash DISPATCHES on it; this call
    site never chooses an algorithm. Every early-return path below (already
    cached, partial already complete) verifies before returning — a shortcut
    that skips the hash is how a corrupt blob becomes permanent.

    Caller runs this in a worker thread; the transfer itself uses the same
    requests stack as the Tensorhub control plane fallback path.

    ``writer_id`` must be unique per concurrent writer (distinct calls, and
    critically distinct PROCESSES when ``dst``'s CAS root is a network volume
    shared by several pods) so two writers of the same missing blob never
    stage into the same temp path. Falls back to a fresh one-off id when not
    supplied, which loses cross-retry resume but is still collision-safe.
    """
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
            verify_file_digest(dst, expected_digest)
            log.info("download_cached path=%s size=%s", dst.name, _human_size(dst.stat().st_size))
            return
        except Exception:
            pass  # re-download

    if not writer_id:
        writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    tmp = dst.parent / f".{dst.name}.part-{writer_id}"

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
        try:
            verify_file_digest(tmp, expected_digest)
        except DigestMismatch:
            log.warning("partial_corrupt path=%s (digest mismatch, restarting)", dst.name)
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
        log.info("download_start path=%s size=%s digest=%s",
                 dst.name, _human_size(expected_size) if expected_size else "unknown",
                 expected_digest[:24])

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

    try:
        verify_file_digest(tmp, expected_digest)
    except DigestMismatch:
        log.error("download_digest_mismatch path=%s expected=%s",
                  dst.name, expected_digest[:24])
        tmp.unlink(missing_ok=True)
        # Propagate the TYPE. DigestMismatch is a ValueError, so the caller's
        # retry loop still gives a flaky transfer its strikes, and the final
        # failure still tells the caller WHICH kind of failure it was.
        raise

    # Durable atomic finalize (gw#408): rename is only atomic in the
    # NAMESPACE — a pod hard-kill after the rename but before writeback
    # persisted a complete-looking blob with truncated/zero data pages,
    # which then poisoned every snapshot built from it. fsync data before
    # the rename and the directory entry after.
    fsync_file(tmp)
    tmp.replace(dst)
    fsync_dir(dst.parent)
    log.info("download_done path=%s size=%s", dst.name, _human_size(actual_size))


def fsync_file(path: Path) -> None:
    """Force a file's data to stable storage (read-only fd works on Linux)."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """Persist a directory entry (the rename itself) to stable storage."""

    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


