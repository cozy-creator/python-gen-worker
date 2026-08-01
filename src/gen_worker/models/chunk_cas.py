"""Chunked sha256 CAS: prefix-dispatched verification and chunk reassembly (pgw#781 / th#1303).

DATA PLANE. Every byte here is fetched, hashed and written by the process that
calls these functions, so under pgw#763 Layer 1 this module belongs to the
COMPUTE CHILD. It deliberately has no IPC, no protocol and no parent handle:
the parent carries control and progress only, and chunk bytes must never be
routed through it. Keep it that way — the module imports nothing from the
worker's transport.

The download shape is **positional materialisation** (th#1362 item 1): the
destination is preallocated and every chunk is streamed straight to ITS OWN
byte range by a bounded worker pool. Retiring safetensors sharding means a
component arrives as ONE multi-GB file, so this function is where everything
sharding used to buy (parallel transfer, resume, partial-failure refetch) has
to actually live.

*   ``K`` chunks are fetched concurrently and each worker ``pwrite``s its blocks
    at ``offset_i`` as they arrive, hashing them on the way past. A chunk counts
    as DONE only once its sha256 matches; until then its range is nobody else's
    business, so a bad attempt is simply overwritten by the retry. RAM per
    worker is one 4 MiB read block, not one 64 MiB chunk — so the window is
    bounded by the pool, not by memory, which is what lets it be wide.
*   The whole-file sha256 runs on its OWN thread over the CONTIGUOUS verified
    prefix, advancing as chunks land. ``hashlib`` drops the GIL, so it overlaps
    the transfer and the file is hashed by about the time it finishes arriving —
    pgw#769's fused check, preserved without an in-order commit bottleneck. The
    whole-file hash is not store-enforced (only the chunks are), so this is
    where that gap closes, and it closes FAIL-CLOSED.
*   **Resume is per chunk and out of order**, recorded in a sidecar journal next
    to the part file. The journal is a HINT, never a durability claim: on
    restart every range it names is re-hashed off disk before it is trusted, so
    an unsynced write that never landed is simply refetched. That is why there
    is no fsync per chunk — the old in-order scheme needed one to make its
    "committed prefix" claim true, and it cost a sync per 64 MiB.
*   pgw#786 is solved PER CHUNK: every chunk fetch carries its own
    :class:`ProgressFloor`, so a source trickling 4 MiB per retry is abandoned
    and refetched on a fresh connection instead of holding a 35 GB file hostage.
    Only when the whole route is bad does the aggregate floor raise a typed
    stall for the hub to re-place the pod.
*   Disk is proven UP FRONT with ``posix_fallocate``, so a 2 TB file that will
    not fit fails in milliseconds instead of 90% of the way through.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Sequence

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
# th#1362: with positional writes a worker holds one 4 MiB read block instead of
# a whole 64 MiB chunk, so the window is priced in threads and sockets rather
# than in gigabytes. 16 is where the measured curve flattens (see
# tests/test_chunk_cas_parallel_th1362.py) — past it the connection pool and the
# GIL give the bytes back.
_DEFAULT_WINDOW = 16


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
    read the algorithm off the digest, never assumed it.

    th#1303 S1: ``blake3`` is no longer hashable here. `parse_cas_ref` still
    RECOGNISES a blake3 ref (that grammar dies at S2, with the shared
    cross-language vectors), so a v1 ref reaching a verifier lands here and
    RAISES. That asymmetry is deliberate: after the repoint no tag resolves to
    a v1 manifest, so the only way to arrive with one is a stale pointer, and
    the safe answer to "I cannot check these bytes" is a refusal, never a skip.
    """
    if algo == "sha256":
        return sha256_file(path)
    raise ValueError(f"unsupported hash algorithm {algo!r}")


def verify_file_digest(path: Path, ref: str) -> None:
    """Verify a file against an algorithm-tagged ref, DISPATCHING ON THE PREFIX.

    Hardcoding the algorithm per call site — which is what the 198 blake3
    references did — is how a sha256 digest gets checked with blake3 and every
    honest file looks corrupt. An empty or undigestable ref RAISES out of
    `parse_cas_ref`; there is no "nothing to check" path through this function.
    """
    algo, want = parse_cas_ref(ref)
    got = hash_file(path, algo)
    if got.lower() != want:
        raise DigestMismatch(
            f"{path.name}: {algo} of bytes is {got[:16]}…, manifest says {want[:16]}…"
        )


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    """``os.pwrite`` is allowed to write short. Positional writes are the whole
    correctness story here, so the loop is not optional."""
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        if n <= 0:
            raise OSError(f"pwrite wrote {n} bytes at offset {offset}")
        view = view[n:]
        offset += n


def _fetch_chunk_to_offset(
    session: requests.Session,
    spec: ChunkSpec,
    fd: int,
    offset: int,
    *,
    on_bytes: Optional[Callable[[int], None]],
) -> None:
    """Stream ONE chunk straight into its byte range, verifying as it goes.

    Abandon-and-refetch is per attempt and per chunk (pgw#786): an attempt that
    stops clearing its own progress floor is dropped and retried on a FRESH
    connection, so one bad connection cannot hold the file.

    The chunk is verified AFTER it lands rather than before, which is safe
    precisely because the range belongs to this chunk alone: an attempt that
    fails verification is overwritten by the retry, and nothing reads the range
    until the caller marks the chunk done. That is the trade that drops RAM per
    worker from a whole 64 MiB chunk to one read block.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _CHUNK_MAX_ATTEMPTS + 1):
        floor = ProgressFloor(_CHUNK_PROGRESS_FLOOR_BYTES)
        delivered = 0
        hasher = hashlib.sha256()
        pos = offset
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
                    delivered += len(block)
                    if delivered > spec.length:
                        raise DigestMismatch(
                            f"chunk {spec.sha256[:12]}: source sent more than {spec.length} bytes"
                        )
                    _pwrite_all(fd, block, pos)
                    pos += len(block)
                    hasher.update(block)
                    if on_bytes is not None:
                        on_bytes(len(block))
            if delivered != spec.length:
                raise DigestMismatch(
                    f"chunk {spec.sha256[:12]}: got {delivered} bytes, manifest says {spec.length}"
                )
            got = hasher.hexdigest()
            if got != spec.sha256:
                # With store-enforced writes the object at this key provably
                # holds these bytes, so a mismatch is TRANSIT corruption.
                raise DigestMismatch(
                    f"chunk {spec.sha256[:12]}: bytes hash to {got[:12]} (transit corruption)"
                )
            return
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


@contextlib.contextmanager
def _cas_fetch_lock(dst: Path) -> Iterator[None]:
    """pgw#783: an OS ``flock`` on a per-digest lock file so G compute children
    sharing ONE container filesystem dedup a fetch of the same CAS entry to ONE
    download instead of G. The in-process ``threading.Lock`` cannot: the groups
    are separate PROCESSES. The first child to arrive holds the lock and
    fetches; the rest block on ``flock`` and, when it releases, find ``dst``
    already present and skip the download (the caller's existence check).

    Best-effort: a platform without ``fcntl`` yields without a lock (the
    pre-pgw#783 behaviour — correctness held, only egress was duplicated), and
    the writer-unique ``tmp`` path already made concurrent writers
    non-corrupting, so the lock is a COST optimisation, never a safety gate."""
    try:
        import fcntl
    except Exception:
        yield
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dst.parent / f".{dst.name}.casfetch.lock"
    # ACQUIRE — its failures must not be confused with the body's, so they are
    # handled entirely before the yield.
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        # Odd FS / NFS without lockd: proceed unserialised rather than fail.
        if fd is not None:
            os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(fd)


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

    # pgw#783: G compute children share ONE container filesystem, so serialise
    # the fetch of this exact CAS entry across processes — the in-process dedup
    # cannot, the groups are separate processes. The first child fetches under
    # the flock; the rest block, then find dst already present and skip (G x
    # egress -> 1x). The lock is a COST optimisation: the writer-unique tmp path
    # already makes concurrent writers non-corrupting, so a lock we cannot take
    # only costs a duplicate download, never correctness.
    with _cas_fetch_lock(dst):
        if dst.exists() and dst.stat().st_size == total_size:
            # A sibling in this pod already materialised it while we waited.
            _log.info("chunk_cas dedup: %s already present (%d bytes); skipping "
                      "the fetch", dst.name, total_size)
            return
        _download_chunked_locked(
            chunks, dst, want_whole=want_whole, total_size=total_size,
            window=window, on_bytes=on_bytes, session_factory=session_factory,
        )


class _Prefix:
    """Which chunks are verified-on-disk, and how far the CONTIGUOUS verified
    prefix reaches. The whole-file hasher rides this: it may only read bytes
    that some worker has already proven."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._done: set[int] = set()
        self._reach = 0
        self._aborted = False
        self._cv = threading.Condition()

    def mark(self, index: int) -> None:
        with self._cv:
            self._done.add(index)
            while self._reach < self._n and self._reach in self._done:
                self._reach += 1
            self._cv.notify_all()

    def abort(self) -> None:
        with self._cv:
            self._aborted = True
            self._cv.notify_all()

    def done(self) -> set[int]:
        with self._cv:
            return set(self._done)

    def await_index(self, index: int) -> bool:
        """Block until chunk ``index`` is inside the verified prefix. False if
        the download aborted first."""
        with self._cv:
            while self._reach <= index and not self._aborted:
                self._cv.wait()
            return self._reach > index


def _journal_header(want_whole: str, total_size: int) -> str:
    return f"sha256:{want_whole} {total_size}"


def _adopt_partial(
    fd: int,
    chunks: Sequence[ChunkSpec],
    offsets: Sequence[int],
    journal: Path,
    header: str,
    window: int,
) -> set[int]:
    """Re-establish what a previous run left behind.

    The journal names chunks a prior run believed it wrote. It is a HINT: every
    range it claims is re-hashed off disk here, so a write that never reached
    the platter is simply refetched. This is what buys resume WITHOUT an fsync
    per chunk, and it resumes out of order — the old committed-prefix scheme
    could only resume a prefix, and in practice could not resume at all,
    because the part file's name carried a fresh uuid on every call.
    """
    try:
        lines = journal.read_text().splitlines()
    except OSError:
        return set()
    if not lines or lines[0] != header:
        # A different file, size or digest was assembled under this name.
        journal.unlink(missing_ok=True)
        return set()
    claimed = sorted({
        int(ln) for ln in (s.strip() for s in lines[1:])
        if ln.isdigit() and 0 <= int(ln) < len(chunks)
    })
    if not claimed:
        return set()

    def _verify(i: int) -> Optional[int]:
        h = hashlib.sha256()
        pos, remaining = offsets[i], chunks[i].length
        while remaining > 0:
            b = os.pread(fd, min(_READ_CHUNK_BYTES, remaining), pos)
            if not b:
                return None
            h.update(b)
            pos += len(b)
            remaining -= len(b)
        return i if h.hexdigest() == chunks[i].sha256 else None

    with ThreadPoolExecutor(max_workers=max(1, min(window, len(claimed)))) as pool:
        kept = {i for i in pool.map(_verify, claimed) if i is not None}
    _log.info("chunk_resume path=%s journal_claimed=%d verified=%d/%d chunks",
              journal.name, len(claimed), len(kept), len(chunks))
    return kept


def _hash_verified_prefix(
    fd: int,
    chunks: Sequence[ChunkSpec],
    offsets: Sequence[int],
    prefix: "_Prefix",
    out: Dict[str, str],
) -> None:
    """Hash the whole file on its own thread, following the verified prefix.

    ``hashlib`` releases the GIL, so this overlaps the transfer: by the time the
    last chunk lands the digest is essentially ready. It re-reads bytes that were
    just written — from page cache in the normal case — which is the price of
    dropping the in-order commit that used to serialise every worker behind one
    file handle.
    """
    h = hashlib.sha256()
    for i in range(len(chunks)):
        if not prefix.await_index(i):
            return
        pos, remaining = offsets[i], chunks[i].length
        while remaining > 0:
            b = os.pread(fd, min(_READ_CHUNK_BYTES, remaining), pos)
            if not b:
                raise OSError(f"short read hashing chunk {i} at offset {pos}")
            h.update(b)
            pos += len(b)
            remaining -= len(b)
    out["digest"] = h.hexdigest()


def _preallocate(fd: int, total_size: int, where: Path) -> None:
    """Prove the disk NOW. A 2 TB single file that will not fit must fail in
    milliseconds, not 90% of the way through the transfer."""
    if os.fstat(fd).st_size != total_size:
        os.ftruncate(fd, total_size)
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is None:
        return
    try:
        fallocate(fd, 0, total_size)
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:  # ENOSPC
            raise InsufficientDiskError(
                f"insufficient disk to allocate {total_size} bytes for {where.name}",
                path=str(where.parent),
            ) from exc
        # EOPNOTSUPP and friends: the ftruncate above still gave us the file.


def _size_session_pool(session: requests.Session, window: int) -> None:
    """urllib3's default pool holds 10 connections. Above that a wide window
    spends its time discarding and re-establishing sockets — measured as a
    THROUGHPUT REGRESSION at window 16 and 32 before this was fixed."""
    try:
        from requests.adapters import HTTPAdapter

        adapter = HTTPAdapter(pool_connections=max(10, window),
                              pool_maxsize=max(10, window))
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:  # noqa: BLE001 - a caller-supplied session may not mount
        pass


def _download_chunked_locked(
    chunks: Sequence[ChunkSpec],
    dst: Path,
    *,
    want_whole: str,
    total_size: int,
    window: int,
    on_bytes: Optional[Callable[[int], None]],
    session_factory: Callable[[], requests.Session],
) -> None:
    """The reassembly itself, run while holding the cross-process CAS fetch lock
    (``download_chunked_file`` does the shape checks and the dedup)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The part file name is STABLE, which is what makes resume reachable at all.
    # It is safe because writers of one CAS entry write IDENTICAL bytes to
    # IDENTICAL offsets — positional writes make concurrent assembly of the same
    # digest convergent rather than corrupting, where the old append stream had
    # to hide behind a per-call uuid (and thereby never resumed).
    tmp = dst.parent / f".{dst.name}.chunkpart"
    journal = dst.parent / f".{dst.name}.chunkdone"
    header = _journal_header(want_whole, total_size)

    offsets: list[int] = []
    running = 0
    for c in chunks:
        offsets.append(running)
        running += c.length

    max_inflight = max(1, min(window, len(chunks)))
    fd = os.open(str(tmp), os.O_RDWR | os.O_CREAT, 0o644)
    jfd: Optional[int] = None
    try:
        _preallocate(fd, total_size, tmp)
        already = _adopt_partial(fd, chunks, offsets, journal, header, max_inflight)

        prefix = _Prefix(len(chunks))
        for i in sorted(already):
            prefix.mark(i)

        # Rewrite the journal from what actually verified, so a false claim is
        # never inherited twice.
        with open(journal, "w") as jf:
            jf.write(header + "\n")
            for i in sorted(already):
                jf.write(f"{i}\n")
        jfd = os.open(str(journal), os.O_WRONLY | os.O_APPEND)

        hashed: Dict[str, str] = {}
        hasher = threading.Thread(
            target=_hash_verified_prefix,
            args=(fd, chunks, offsets, prefix, hashed),
            daemon=True,
        )
        hasher.start()
        try:
            _fetch_pending(
                chunks, offsets, prefix, jfd, fd, dst,
                already=already, max_inflight=max_inflight,
                on_bytes=on_bytes, session_factory=session_factory,
            )
        except BaseException:
            prefix.abort()
            hasher.join(timeout=30)
            raise
        hasher.join()
        got = hashed.get("digest")
        if got is None:
            raise OSError(f"{dst.name}: whole-file hash did not complete")
        _finalize(fd, tmp, journal, dst, got=got, want_whole=want_whole,
                  total_size=total_size)
    finally:
        if jfd is not None:
            os.close(jfd)
        os.close(fd)


def _fetch_pending(
    chunks: Sequence[ChunkSpec],
    offsets: Sequence[int],
    prefix: "_Prefix",
    jfd: int,
    fd: int,
    dst: Path,
    *,
    already: set[int],
    max_inflight: int,
    on_bytes: Optional[Callable[[int], None]],
    session_factory: Callable[[], requests.Session],
) -> None:
    pending = [i for i in range(len(chunks)) if i not in already]
    if not pending:
        return

    aggregate = ProgressFloor(_CHUNK_PROGRESS_FLOOR_BYTES * max_inflight)
    delivered_total = 0
    counter = threading.Lock()

    def _count(n: int) -> None:
        nonlocal delivered_total
        with counter:
            delivered_total += n
        if on_bytes is not None:
            on_bytes(n)

    session = session_factory()
    _size_session_pool(session, max_inflight)

    def _one(i: int) -> None:
        _fetch_chunk_to_offset(session, chunks[i], fd, offsets[i], on_bytes=_count)
        # O_APPEND makes a short line atomic across threads, so the journal
        # needs no lock of its own.
        os.write(jfd, f"{i}\n".encode())
        prefix.mark(i)

    try:
        with ThreadPoolExecutor(max_workers=max_inflight) as pool:
            futures: Dict["Future[None]", int] = {
                pool.submit(_one, i): i for i in pending
            }
            failure: Optional[tuple[int, BaseException]] = None
            for fut in as_completed(futures):
                try:
                    fut.result()
                except BaseException as exc:  # noqa: BLE001 - classified below
                    failure = (futures[fut], exc)
                    del exc
                    for other in futures:
                        other.cancel()
                    break
        if failure is not None:
            index, cause = failure
            if isinstance(cause, (UrlExpiredError, DigestMismatch)):
                raise cause
            if getattr(cause, "errno", None) == 28:  # ENOSPC
                raise InsufficientDiskError(
                    f"insufficient disk assembling {dst.name}",
                    path=str(dst.parent),
                ) from cause
            if not aggregate.cleared(delivered_total):
                # Nothing anywhere is moving: the ROUTE is bad, not one
                # connection. Typed so the hub re-places the pod.
                raise ChunkedDownloadStalled(
                    f"{dst.name}: chunk {index} failed and no chunk "
                    f"cleared the aggregate floor ({delivered_total} bytes "
                    f"delivered) — download_stalled"
                ) from cause
            raise cause
    finally:
        session.close()


def _finalize(
    fd: int,
    tmp: Path,
    journal: Path,
    dst: Path,
    *,
    got: str,
    want_whole: str,
    total_size: int,
) -> None:
    actual = os.fstat(fd).st_size
    if actual != total_size:
        tmp.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        raise DigestMismatch(
            f"{dst.name}: assembled {actual} bytes, manifest says {total_size}"
        )
    if got != want_whole:
        # FAIL-CLOSED. The whole-file hash is NOT store-enforced (only the
        # chunks are), so this is the one place a lying manifest label is
        # caught — and it must never install the file anyway. Every chunk
        # verified individually and landed at an offset the manifest itself
        # dictated, so the bytes on disk ARE the manifest's file: what is wrong
        # is the label, or the chunk list it labels.
        tmp.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        raise DigestMismatch(
            f"{dst.name}: reassembled bytes hash to {got[:16]}…, manifest says "
            f"{want_whole[:16]}… (chunks were individually valid, so the "
            f"whole-file label is wrong or the chunk ORDER is)"
        )
    # Durable atomic finalize (gw#408): rename is atomic in the NAMESPACE only.
    from .cozy_cas import fsync_dir

    os.fsync(fd)
    tmp.replace(dst)
    fsync_dir(dst.parent)
    journal.unlink(missing_ok=True)
    _log.info("chunk_download_done path=%s size=%d sha256=%s", dst.name, total_size, got[:16])
