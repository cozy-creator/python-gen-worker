"""Chunked sha256 CAS: prefix-dispatched verification and chunk reassembly (pgw#781 / th#1303).

DATA PLANE. Every byte here is fetched, hashed and written by the process that
calls these functions, so under pgw#763 Layer 1 this module belongs to the
COMPUTE CHILD. It deliberately has no IPC, no protocol and no parent handle:
the parent carries control and progress only, and chunk bytes must never be
routed through it. Keep it that way — the module imports nothing from the
worker's transport.

The download shape is **bounded out-of-order fetch, in-order commit**:

*   A window of ``K`` chunks is fetched concurrently. Each arrives in RAM and is
    sha256-verified BEFORE it is committed, so a corrupt chunk is discarded and
    refetched rather than written.
*   Chunks are committed to the destination **in order**, and the commit stream
    feeds the whole-file hasher. The manifest's whole-file sha256 is therefore
    verified in the SAME pass that assembles the file — pgw#769's fused-check
    requirement — with no second read of the bytes. The whole-file hash is not
    store-enforced (only the chunks are), so this is where that gap closes, and
    it closes FAIL-CLOSED: a mismatch deletes and refetches, then raises typed.
*   The committed prefix IS the crash-resume floor. A restart re-fetches at most
    the in-flight window, because a partial file of length L means chunks
    ``0..L/chunk_size`` are already durable.
*   pgw#786 is solved PER CHUNK: every chunk fetch carries its own
    :class:`ProgressFloor`, so a source trickling 4 MiB per retry is abandoned
    and refetched on a fresh connection instead of holding a 35 GB file hostage.
    Only when the whole route is bad does the aggregate floor raise a typed
    stall for the hub to re-place the pod.

Why the window is small: ``K`` chunks of 64 MiB are held in RAM at once, so
K=6 is ~384 MiB. It is scaled down against the cgroup budget rather than the
host's total RAM (a pod's memory.max is what kills it).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import requests

from ..capability import InsufficientDiskError
from ..stall import ProgressFloor
from .errors import UrlExpiredError

__all__ = [
    "CAS_CHUNK_SIZE_BYTES",
    "ChunkSpec",
    "ChunkedDownloadStalled",
    "DigestMismatch",
    "chunk_count_for",
    "chunk_len_at",
    "download_chunked_file",
    "hash_file",
    "parse_cas_ref",
    "sha256_file",
    "verify_file_digest",
]

_log = logging.getLogger(__name__)

# Must equal storage.CASChunkSizeBytes on the hub. A disagreement does not
# corrupt anything (the manifest carries every chunk's length explicitly) but
# it does break the resume arithmetic, so the value is asserted against the
# manifest's own chunk_size_bytes on every download.
CAS_CHUNK_SIZE_BYTES = 64 * 1024 * 1024

_READ_CHUNK_BYTES = 4 * 1024 * 1024

# Per-chunk watchdog. A chunk's size is known a priori, so unlike a whole-file
# transfer we can say what "too slow" means: a healthy host sustains >=100 MB/s
# on ranged GETs (pgw#786 measured 250 MB/s), so 4 MiB per window is two orders
# of magnitude below healthy and unambiguously a lemon.
_CHUNK_PROGRESS_FLOOR_BYTES = 4 * 1024 * 1024
_CHUNK_MAX_ATTEMPTS = 6
_DEFAULT_WINDOW = 6


class DigestMismatch(ValueError):
    """Bytes did not hash to the digest they were fetched under."""


class ChunkedDownloadStalled(RuntimeError):
    """Every chunk source is below the progress floor — the ROUTE is bad.

    Typed so the hub can re-place the pod rather than retrying into the same
    lemon host (pgw#786).
    """


@dataclass(frozen=True)
class ChunkSpec:
    """One CAS object of a chunked file.

    ``sha256`` is bare lowercase hex; ``length`` is carried explicitly by the
    manifest so a chunk's cumulative offset is known without assuming the
    chunking policy.
    """

    sha256: str
    url: str
    length: int


def parse_cas_ref(ref: str) -> tuple[str, str]:
    """Split ``"<algo>:<hex>"`` into its parts.

    A BARE hex string is read as legacy blake3 — matching the hub's read-path
    rule — because pre-migration manifests legitimately carry bare hex. New
    code must always emit the tagged form; bare hex is refused at phase 4.
    """
    s = (ref or "").strip().lower()
    if not s:
        raise ValueError("cas ref: empty")
    if ":" in s:
        algo, _, hexpart = s.partition(":")
        algo, hexpart = algo.strip(), hexpart.strip()
    else:
        algo, hexpart = "blake3", s
    if algo not in ("sha256", "blake3"):
        raise ValueError(f"cas ref: unsupported algorithm {algo!r}")
    if len(hexpart) != 64 or any(c not in "0123456789abcdef" for c in hexpart):
        raise ValueError(f"cas ref: {algo} digest must be 64 hex characters")
    return algo, hexpart


def chunk_count_for(size: int, chunk_size: int = CAS_CHUNK_SIZE_BYTES) -> int:
    """How many chunks a file of ``size`` bytes is stored as; 0 = stored whole."""
    if size <= chunk_size:
        return 0
    return (size + chunk_size - 1) // chunk_size


def chunk_len_at(size: int, index: int, chunk_size: int = CAS_CHUNK_SIZE_BYTES) -> int:
    off = index * chunk_size
    if off >= size:
        return 0
    return min(chunk_size, size - off)


def sha256_file(path: Path, chunk_size: int = _READ_CHUNK_BYTES) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_file(path: Path, algo: str) -> str:
    """Hash a file with the named algorithm. No default: the caller must have
    read the algorithm off the digest, never assumed it."""
    if algo == "sha256":
        return sha256_file(path)
    if algo == "blake3":
        from blake3 import blake3  # local: the dep leaves at phase 4

        h = blake3(max_threads=blake3.AUTO)
        with open(path, "rb") as f:
            while True:
                b = f.read(_READ_CHUNK_BYTES)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    raise ValueError(f"unsupported hash algorithm {algo!r}")


def verify_file_digest(path: Path, ref: str) -> None:
    """Verify a file against an algorithm-tagged ref, DISPATCHING ON THE PREFIX.

    This is the whole dual-read window on the worker side. Hardcoding the
    algorithm per call site — which is what the 198 blake3 references did — is
    how a sha256 digest gets checked with blake3 and every honest file looks
    corrupt.
    """
    algo, want = parse_cas_ref(ref)
    got = hash_file(path, algo)
    if got.lower() != want:
        raise DigestMismatch(
            f"{path.name}: {algo} of bytes is {got[:16]}…, manifest says {want[:16]}…"
        )


def _fetch_chunk_bytes(
    session: requests.Session,
    spec: ChunkSpec,
    *,
    on_bytes: Optional[Callable[[int], None]],
) -> bytes:
    """Fetch ONE chunk into RAM, verify its sha256, return the bytes.

    Abandon-and-refetch is per attempt and per chunk (pgw#786): an attempt that
    stops clearing its own progress floor is dropped and retried on a FRESH
    connection, so one bad connection cannot hold the file. The chunk is
    verified in RAM, before it can be committed.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _CHUNK_MAX_ATTEMPTS + 1):
        floor = ProgressFloor(_CHUNK_PROGRESS_FLOOR_BYTES)
        delivered = 0
        buf = bytearray()
        try:
            with session.get(spec.url, stream=True, timeout=(60, 180)) as resp:
                sc = int(resp.status_code)
                if 400 <= sc < 500 and sc not in (408, 429):
                    raise UrlExpiredError(
                        f"chunk URL rejected with HTTP {sc}", status_code=sc
                    )
                resp.raise_for_status()
                for block in resp.iter_content(chunk_size=_READ_CHUNK_BYTES):
                    if not block:
                        continue
                    buf.extend(block)
                    delivered += len(block)
                    if on_bytes is not None:
                        on_bytes(len(block))
                    if len(buf) > spec.length:
                        raise DigestMismatch(
                            f"chunk {spec.sha256[:12]}: source sent more than {spec.length} bytes"
                        )
            if len(buf) != spec.length:
                raise DigestMismatch(
                    f"chunk {spec.sha256[:12]}: got {len(buf)} bytes, manifest says {spec.length}"
                )
            got = hashlib.sha256(buf).hexdigest()
            if got != spec.sha256:
                # With store-enforced writes the object at this key provably
                # holds these bytes, so a mismatch is TRANSIT corruption.
                raise DigestMismatch(
                    f"chunk {spec.sha256[:12]}: bytes hash to {got[:12]} (transit corruption)"
                )
            return bytes(buf)
        except UrlExpiredError:
            raise
        except (requests.RequestException, DigestMismatch, OSError) as exc:
            last_exc = exc
            stalled = not floor.cleared(delivered)
            _log.warning(
                "chunk_refetch sha256=%s attempt=%d/%d delivered=%d stalled=%s: %s",
                spec.sha256[:12], attempt, _CHUNK_MAX_ATTEMPTS, delivered, stalled, exc,
            )
            # No sleep and no backoff on a STALL: the point is to abandon this
            # connection immediately and open a new one. Real errors fall
            # through to the same fresh-connection retry.
            continue
    assert last_exc is not None
    raise last_exc


def download_chunked_file(
    chunks: Sequence[ChunkSpec],
    dst: Path,
    *,
    whole_digest: str,
    total_size: int,
    chunk_size_bytes: int = CAS_CHUNK_SIZE_BYTES,
    window: int = _DEFAULT_WINDOW,
    on_bytes: Optional[Callable[[int], None]] = None,
    session_factory: Callable[[], requests.Session] = requests.Session,
) -> None:
    """Reassemble a chunked file, verifying every chunk AND the whole-file hash.

    The whole-file hash is fused into the commit stream: there is exactly one
    pass over the bytes, and no second read. On mismatch the partial file is
    deleted so the next attempt starts clean.
    """
    if not chunks:
        raise ValueError("download_chunked_file: no chunks")
    algo, want_whole = parse_cas_ref(whole_digest)
    if algo != "sha256":
        raise ValueError(f"chunked files are sha256-only, got {algo!r}")

    # ---- shape checks BEFORE any byte moves ----
    declared = sum(c.length for c in chunks)
    if declared != total_size:
        raise ValueError(
            f"chunk lengths sum to {declared}, manifest says {total_size}"
        )
    expect_n = chunk_count_for(total_size, chunk_size_bytes)
    if expect_n != len(chunks):
        raise ValueError(
            f"{len(chunks)} chunks declared, size {total_size} at chunk size "
            f"{chunk_size_bytes} needs {expect_n}"
        )
    for i, c in enumerate(chunks):
        parse_cas_ref("sha256:" + c.sha256)
        want_len = chunk_len_at(total_size, i, chunk_size_bytes)
        if c.length != want_len:
            raise ValueError(
                f"chunk {i} length {c.length}, fixed chunking requires {want_len}"
            )
        if not (c.url or "").strip():
            raise ValueError(f"chunk {i} has no URL")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Writer-unique, stable for this call so its own resume works, but distinct
    # from every other writer of the same digest — including another PROCESS
    # sharing a network CAS volume (th#850).
    writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    tmp = dst.parent / f".{dst.name}.chunkpart-{writer_id}"

    # ---- crash-resume: the committed prefix is the floor ----
    committed = 0
    start_index = 0
    if tmp.exists():
        have = tmp.stat().st_size
        # Only a WHOLE number of chunks is trusted as durable: a torn tail
        # belongs to a chunk whose commit did not finish.
        while start_index < len(chunks) and committed + chunks[start_index].length <= have:
            committed += chunks[start_index].length
            start_index += 1
        if committed != have:
            # Truncate the torn tail rather than re-fetching everything.
            with open(tmp, "r+b") as f:
                f.truncate(committed)
        if start_index:
            _log.info(
                "chunk_resume path=%s committed=%d/%d chunks (%d bytes)",
                dst.name, start_index, len(chunks), committed,
            )

    hasher = hashlib.sha256()
    if start_index:
        # The resumed prefix must be re-read to seed the hasher. This is the
        # ONLY case where bytes are read twice, and it is bounded by what a
        # previous process already committed — never by the whole file on a
        # fresh download.
        with open(tmp, "rb") as seed:
            remaining = committed
            while remaining > 0:
                b = seed.read(min(_READ_CHUNK_BYTES, remaining))
                if not b:
                    raise OSError(f"{tmp}: short read while seeding the resume hash")
                hasher.update(b)
                remaining -= len(b)

    pending = list(range(start_index, len(chunks)))
    if not pending:
        _finalize(tmp, dst, hasher, want_whole, total_size)
        return

    # ---- bounded out-of-order fetch, in-order commit ----
    max_inflight = max(1, min(window, len(pending)))
    ready: Dict[int, bytes] = {}
    next_commit = start_index
    aggregate = ProgressFloor(_CHUNK_PROGRESS_FLOOR_BYTES * max_inflight)
    delivered_total = 0
    lock = threading.Lock()

    def _count(n: int) -> None:
        nonlocal delivered_total
        with lock:
            delivered_total += n
        if on_bytes is not None:
            on_bytes(n)

    session = session_factory()
    try:
        with open(tmp, "ab") as out, ThreadPoolExecutor(max_workers=max_inflight) as pool:
            inflight: Dict[int, "Future[bytes]"] = {}
            cursor = start_index

            def _submit_until_full() -> None:
                nonlocal cursor
                while len(inflight) < max_inflight and cursor < len(chunks):
                    idx = cursor
                    inflight[idx] = pool.submit(
                        _fetch_chunk_bytes, session, chunks[idx], on_bytes=_count
                    )
                    cursor += 1

            _submit_until_full()
            while next_commit < len(chunks):
                fut = inflight.pop(next_commit)
                try:
                    data = fut.result()
                except UrlExpiredError:
                    raise
                except DigestMismatch:
                    raise
                except Exception as exc:  # noqa: BLE001 - classified below
                    if getattr(exc, "errno", None) == 28:  # ENOSPC
                        raise InsufficientDiskError(
                            f"insufficient disk assembling {dst.name}",
                            path=str(dst.parent),
                        ) from exc
                    if not aggregate.cleared(delivered_total):
                        # Nothing anywhere is moving: the ROUTE is bad, not one
                        # connection. Typed so the hub re-places the pod.
                        raise ChunkedDownloadStalled(
                            f"{dst.name}: chunk {next_commit} failed and no chunk "
                            f"cleared the aggregate floor ({delivered_total} bytes "
                            f"delivered) — download_stalled"
                        ) from exc
                    raise
                ready[next_commit] = data
                # Commit in order, feeding the whole-file hasher from the SAME
                # bytes that reach the disk.
                while next_commit in ready:
                    block = ready.pop(next_commit)
                    out.write(block)
                    hasher.update(block)
                    next_commit += 1
                out.flush()
                os.fsync(out.fileno())
                _submit_until_full()
    finally:
        session.close()

    _finalize(tmp, dst, hasher, want_whole, total_size)


def _finalize(
    tmp: Path, dst: Path, hasher: "hashlib._Hash", want_whole: str, total_size: int
) -> None:
    actual = tmp.stat().st_size
    if actual != total_size:
        tmp.unlink(missing_ok=True)
        raise DigestMismatch(
            f"{dst.name}: assembled {actual} bytes, manifest says {total_size}"
        )
    got = hasher.hexdigest()
    if got != want_whole:
        # FAIL-CLOSED. The whole-file hash is NOT store-enforced (only the
        # chunks are), so this is the one place a lying manifest label is
        # caught — and it must never install the file anyway.
        tmp.unlink(missing_ok=True)
        raise DigestMismatch(
            f"{dst.name}: reassembled bytes hash to {got[:16]}…, manifest says "
            f"{want_whole[:16]}… (chunks were individually valid, so the "
            f"whole-file label is wrong or the chunk ORDER is)"
        )
    # Durable atomic finalize (gw#408): rename is atomic in the NAMESPACE only.
    from .cozy_cas import fsync_dir, fsync_file

    fsync_file(tmp)
    tmp.replace(dst)
    fsync_dir(dst.parent)
    _log.info("chunk_download_done path=%s size=%d sha256=%s", dst.name, total_size, got[:16])
