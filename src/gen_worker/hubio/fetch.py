"""``fetch_verified`` — THE single-object verified download.

One place for the whole discipline:

- cache short-circuit (a verified ``dst`` is never re-fetched);
- writer-unique ``.part`` staging (two writers of one blob, including two PODS
  on a shared network volume, never share a temp file);
- HTTP-Range resume with restart-on-ignored-Range;
- the byte cap INSIDE the stream loop (``expected_size`` when the manifest
  declares one, ``cap_bytes`` as the caller's bound of last resort; a breach
  raises :class:`~gen_worker.bounded_stream.StreamTooLarge`);
- algorithm-tagged digest verification, dispatching on the tag;
- durable atomic finalize (fsync data, rename, fsync directory);
- progress-floor retry (strikes count CONSECUTIVE attempts that fail to move
  the byte floor — a link that keeps landing bytes never times out).

NOT this primitive: the chunk-CAS positional engine (many chunks into one
file with cross-pod adopt+rehash — ``models/chunk_cas``), the SSRF-guarded
input-asset fetch (``input_assets`` applies the guarded opener per hop), and
the hf_hub-owned HF branch.
"""

from __future__ import annotations

import errno
import logging
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from ..bounded_stream import StreamTooLarge
from ..capability import InsufficientDiskError
from ..models.chunk_cas import DigestMismatch, parse_cas_ref, verify_file_digest
from ..models.errors import UrlExpiredError
from ..stall import ProgressFloor

_log = logging.getLogger("gen_worker.download")

_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024

# Retry policy. Verification failures (size/digest mismatch) mean the source
# blob is likely corrupt — 2 re-downloads, then give up. UrlExpiredError /
# ENOSPC never retry.
#
# The transient half gives up on PROGRESS, never on a wall clock: a clock cannot
# tell a 200GB blob crawling in over a bad pod uplink from a wedge, and
# resume-on-retry means the slow case is genuinely converging.
# `_TRANSIENT_MAX_TRIES` counts CONSECUTIVE transient failures that failed to
# move the byte floor; any attempt delivering `_RETRY_PROGRESS_FLOOR_BYTES`
# resets the count.
_TRANSIENT_MAX_TRIES = 30
_RETRY_PROGRESS_FLOOR_BYTES = 8 * 1024 * 1024
_VERIFY_MAX_FAILURES = 3  # initial try + 2 retries
_RETRY_BACKOFF_CAP_S = 30.0

#: Algorithms servable today: the hub content-addresses by sha256; a caller
#: with a wider live vocabulary widens this explicitly.
DEFAULT_ALLOWED_ALGOS: tuple[str, ...] = ("sha256",)


def _check_status(resp: requests.Response) -> None:
    """4xx (except 408/429) on a presigned URL is dead — no retry can help."""
    sc = int(resp.status_code)
    if 400 <= sc < 500 and sc not in (408, 429):
        raise UrlExpiredError(f"download URL rejected with HTTP {sc}", status_code=sc)
    resp.raise_for_status()


def _human_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}KB"
    return f"{n}B"


def fetch_once(
    url: str,
    dst: Path,
    *,
    expected_digest: str,
    expected_size: int = 0,
    cap_bytes: int = 0,
    on_bytes: Optional[Callable[[int], None]] = None,
    writer_id: Optional[str] = None,
    resume: bool = True,
) -> None:
    """ONE verified fetch attempt: Range resume, in-loop cap, digest verify,
    durable atomic rename. Callers own retry (or use :func:`fetch_verified`).

    ``resume=False`` disables partial retention entirely: the temp file is
    unlinked on ANY failure (the loser cleans up after itself — the dataset
    contract), where the CAS callers instead keep partials for Range resume
    across retries.

    ``writer_id`` must be unique per concurrent writer (distinct calls, and
    critically distinct PROCESSES when ``dst`` sits on a network volume
    shared by several pods) so two writers of the same missing blob never
    stage into the same temp path. Stable across one caller's retries, so
    HTTP-Range resume still works; falls back to a fresh one-off id, which
    loses cross-retry resume but stays collision-safe.
    """
    log = _log

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
    if resume and tmp.exists():
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

    # The in-loop bound: the declared size when the manifest has one, the
    # caller's cap of last resort otherwise. Zero = the caller explicitly
    # declined a fallback bound and the declaration is the bound.
    limit = int(expected_size) or int(cap_bytes)

    def _stream(resp: requests.Response, *, write_mode: str, start: int) -> None:
        downloaded = start
        last_log = start
        log_every = max(expected_size // 10, 50 << 20) if expected_size else (100 << 20)
        with open(tmp, write_mode) as f:
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if limit and downloaded > limit:
                    raise StreamTooLarge(dst.name, limit, downloaded)
                f.write(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
                if downloaded - last_log >= log_every:
                    pct = f" ({100 * downloaded // expected_size}%)" if expected_size else ""
                    log.info("download_progress path=%s downloaded=%s%s",
                             dst.name, _human_size(downloaded), pct)
                    last_log = downloaded

    try:
        with requests.Session() as session:
            with session.get(url, headers=headers, stream=True, timeout=(60, 180)) as resp:
                content_range = str(resp.headers.get("Content-Range") or "").strip()

                # Server ignored Range or returned unexpected range start?
                # Restart from byte 0 to avoid corrupted appends.
                if offset and (
                    resp.status_code == 200
                    or (resp.status_code == 206 and not content_range.startswith(f"bytes {offset}-"))
                ):
                    log.info("download_range_ignored path=%s status=%s (restarting from 0)",
                             dst.name, resp.status_code)
                    resp.close()
                    with session.get(url, stream=True, timeout=(60, 180)) as resp2:
                        _check_status(resp2)
                        _stream(resp2, write_mode="wb", start=0)
                else:
                    _check_status(resp)
                    _stream(resp, write_mode=mode, start=offset)
    except StreamTooLarge:
        tmp.unlink(missing_ok=True)
        raise
    except BaseException:
        if not resume:
            tmp.unlink(missing_ok=True)
        raise

    # Validate.
    actual_size = tmp.stat().st_size
    if expected_size and actual_size != expected_size:
        log.error("download_size_mismatch path=%s expected=%s got=%s",
                  dst.name, expected_size, actual_size)
        tmp.unlink(missing_ok=True)
        raise ValueError(f"size mismatch (expected {expected_size}, got {actual_size})")

    try:
        verify_file_digest(tmp, expected_digest)
    except DigestMismatch:
        log.error("download_digest_mismatch path=%s expected=%s",
                  dst.name, expected_digest[:24])
        tmp.unlink(missing_ok=True)
        # Propagate the TYPE. DigestMismatch is a ValueError, so a caller's
        # retry loop still gives a flaky transfer its strikes, and the final
        # failure still tells the caller WHICH kind of failure it was.
        raise

    # Durable atomic finalize: rename is only atomic in the NAMESPACE — a pod
    # hard-kill after the rename but before writeback persists a
    # complete-looking blob with truncated/zero data pages, poisoning every
    # snapshot built from it. fsync data before the rename, the directory entry
    # after.
    from ..models.cozy_cas import fsync_dir, fsync_file  # local: avoids a module cycle

    fsync_file(tmp)
    tmp.replace(dst)
    fsync_dir(dst.parent)
    log.info("download_done path=%s size=%s", dst.name, _human_size(actual_size))


def fetch_verified(
    url: str,
    dst: Path,
    *,
    expected_digest: str,
    expected_size: int = 0,
    cap_bytes: int = 0,
    on_bytes: Optional[Callable[[int], None]] = None,
    allowed_algos: tuple[str, ...] = DEFAULT_ALLOWED_ALGOS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """The retrying verified fetch.

    The digest is ALGORITHM-TAGGED and MANDATORY; an absent, untagged, or
    disallowed-algorithm digest raises before a single byte is fetched —
    a permanent refusal must not spend three retries looking like a flaky
    network. There is no unverified path through this function.
    """
    if not str(expected_digest or "").strip():
        raise ValueError(f"download of {dst.name}: no expected digest — refusing")
    hash_algo, _ = parse_cas_ref(expected_digest)
    if hash_algo not in allowed_algos:
        raise ValueError(
            f"download of {dst.name}: digest algorithm {hash_algo!r} is not "
            f"verifiable here (allowed: {', '.join(allowed_algos)}) — refusing "
            "rather than fetching unchecked"
        )
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

    writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    while True:
        try:
            fetch_once(
                url, dst,
                expected_digest=expected_digest,
                expected_size=expected_size,
                cap_bytes=cap_bytes,
                on_bytes=_count_bytes,
                writer_id=writer_id,
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
            sleep(delay)


async def afetch_verified(
    url: str,
    dst: Path,
    *,
    expected_digest: str,
    expected_size: int = 0,
    cap_bytes: int = 0,
    on_bytes: Optional[Callable[[int], None]] = None,
    allowed_algos: tuple[str, ...] = DEFAULT_ALLOWED_ALGOS,
) -> None:
    """Async wrapper: the whole retrying fetch runs in a worker thread; the
    event loop never blocks on transfer or backoff."""
    import asyncio

    await asyncio.to_thread(
        fetch_verified,
        url, dst,
        expected_digest=expected_digest,
        expected_size=expected_size,
        cap_bytes=cap_bytes,
        on_bytes=on_bytes,
        allowed_algos=allowed_algos,
    )
