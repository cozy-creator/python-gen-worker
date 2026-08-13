"""Chunked sha256 CAS: prefix-dispatched verification and chunk reassembly.

DATA PLANE. Every byte here is fetched, hashed and written by the process that
calls these functions, so this module belongs to the COMPUTE CHILD. It
deliberately has no IPC, no protocol and no parent handle: the parent carries
control and progress only, and chunk bytes must never be routed through it. Keep
it that way — the module imports nothing from the worker's transport.

The download shape is **positional materialisation**: the destination is
preallocated and every chunk is streamed straight to ITS OWN byte range by a
bounded worker pool. A component arrives as ONE multi-GB file, so parallel
transfer, resume and partial-failure refetch all have to live here.

*   ``K`` chunks are fetched concurrently and each worker ``pwrite``s its blocks
    at ``offset_i`` as they arrive, hashing them on the way past. A chunk counts
    as DONE only once its sha256 matches; until then its range is nobody else's
    business, so a bad attempt is simply overwritten by the retry. RAM per
    worker is one 4 MiB read block, not one 64 MiB chunk — so the window is
    bounded by the pool, not by memory, which is what lets it be wide.
*   The whole-file sha256 runs on its OWN thread over the CONTIGUOUS verified
    prefix, advancing as chunks land. ``hashlib`` drops the GIL, so it overlaps
    the transfer and the file is hashed by about the time it finishes arriving,
    without an in-order commit bottleneck. The whole-file hash is not
    store-enforced (only the chunks are), so this is where that gap closes, and
    it closes FAIL-CLOSED.
*   **Resume is per chunk and out of order**, recorded in a sidecar journal next
    to the part file. The journal is a HINT, never a durability claim: on
    restart every range it names is re-hashed off disk before it is trusted, so
    an unsynced write that never landed is simply refetched. That is why there
    is no fsync per chunk.
*   Every chunk fetch carries its own :class:`ProgressFloor`, so a source
    trickling 4 MiB per retry is abandoned and refetched on a fresh connection
    instead of holding a 35 GB file hostage. Only when the whole route is bad
    does the aggregate floor raise a typed stall for the hub to re-place the pod.
*   **Both stores are filled AT ONCE**. When an endpoint volume is
    attached, every block is ``pwrite``-ed to its offset in the volume's part
    file in the same pass as the local one, so the volume write hides behind
    network latency instead of costing a second full read + write + hash of
    every byte after the fetch. The mirror is BEST EFFORT and writer-unique: it
    is published only after the LOCAL file passes its whole-file digest, so a
    half-written volume blob is never readable under the digest name, and any
    mirror failure disables the mirror rather than failing the fetch.
*   **Resume crosses PODS, at chunk granularity**. Each chunk is copied onto
    the volume the moment its own sha256 verifies, under the
    content-addressed name ``chunks/<algo>/aa/bb/<file>/<index>-<chunk>``. So
    the volume never holds unverified bytes under a name anyone reads: it holds
    individually-VERIFIED chunks of an incomplete file. A successor pod builds
    those names straight from the manifest, RE-HASHES every chunk it adopts
    before trusting it, and fetches only what is missing — in any order, since
    a CAS file's chunks have none. The names are content-addressed, so two pods
    writing one chunk write identical bytes to one name and a duplicate write
    is a no-op. Cleanup is targeted and clock-free: whoever establishes that the
    volume holds the COMPLETE blob drops that file's chunk objects.
*   Disk is proven UP FRONT with ``posix_fallocate``, so a 2 TB file that will
    not fit fails in milliseconds instead of 90% of the way through.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import random
import threading
import time
import uuid
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
    "drop_volume_chunks",
    "hash_file",
    "hasher_for",
    "parse_cas_ref",
    "sha256_file",
    "verify_file_digest",
    "volume_chunk_dir",
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
# on ranged GETs (measured 250 MB/s), so 4 MiB per window is two orders of
# magnitude below healthy and unambiguously a lemon.
_CHUNK_PROGRESS_FLOOR_BYTES = 4 * 1024 * 1024
# A chunk's retry budget is a STRIKE COUNT plus a wall-clock cap, and the failure
# shapes are told apart before either is charged.
#
#   * A `DigestMismatch` is a COMPLETE, wrong-hashing body: the route is
#     demonstrably healthy and only the bytes are not, so it refetches at once.
#     Sleeping would only delay the refetch of an object the store is willing
#     to serve.
#   * An attempt that DELIVERED ITS FLOOR and then died proves the route is
#     alive — a lemon CONNECTION. Abandon it and open a fresh socket AT ONCE:
#     no sleep, and the strike count starts over.
#   * An attempt that delivered nothing is the network not being there. It
#     BACKS OFF, jittered `2**n`, as `cozy_cas`'s whole-file loop does.
#
# There is deliberately no third "trickle" case: `delivered` counts blocks
# `iter_content` actually YIELDS, and a connection that dies mid-block yields
# nothing at all, so "some bytes, but under the floor" is unreachable below one
# read block.
#
# The give-up is a COUNT, never a clock (`stall.py`'s standing rule: a wall
# clock cannot tell a healthy slow transfer from a wedge). Clearing the floor
# resets the BACKOFF, not the budget — an attempt that restarts the chunk from
# offset 0 every time has advanced nothing, however many bytes it moved, so
# resetting the budget on it would loop forever. 8 attempts at this backoff is
# ~45-90s of tolerated blackout per chunk, and the executor's 3 outer attempts
# (`_DOWNLOAD_RETRIES`, which the journal resumes across) sit on top of it.
_CHUNK_MAX_ATTEMPTS = 8
_CHUNK_BACKOFF_CAP_S = 30.0
# With positional writes a worker holds one 4 MiB read block instead of a whole
# 64 MiB chunk, so the window is priced in threads and sockets rather than in
# gigabytes. 16 is where the measured curve flattens — past it the connection
# pool and the GIL give the bytes back.
_DEFAULT_WINDOW = 16


class DigestMismatch(ValueError):
    """Bytes did not hash to the digest they were fetched under."""


class ChunkedDownloadStalled(RuntimeError):
    """Every chunk source is below the progress floor — the ROUTE is bad.

    Typed so the hub can re-place the pod rather than retrying into the same
    lemon host.
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

    An UNTAGGED ref is REFUSED — the same rule as the hub's
    ``storage.ParseCASRef``. A 64-character hex cannot distinguish blake3 from
    sha256, because both digests are 32 bytes: the length check is not a
    discriminator, it only looks like one. Inferring an algorithm addresses the
    WRONG namespace silently.
    """
    s = (ref or "").strip().lower()
    if not s:
        raise ValueError("cas ref: empty")
    if ":" not in s:
        raise ValueError(
            'cas ref: not algorithm-tagged (bare hex is refused; write "sha256:<hex>")'
        )
    algo, _, hexpart = s.partition(":")
    algo, hexpart = algo.strip(), hexpart.strip()
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


def hasher_for(algo: str) -> "hashlib._Hash":
    """A fresh incremental hasher for the named algorithm — the same dispatch as
    :func:`hash_file`, for callers that can FUSE the hash into a read they
    already perform rather than pay a second pass over the bytes."""
    if algo == "sha256":
        return hashlib.sha256()
    raise ValueError(f"unsupported hash algorithm {algo!r}")


def hash_file(path: Path, algo: str) -> str:
    """Hash a file with the named algorithm. No default: the caller must have
    read the algorithm off the digest, never assumed it.

    ``blake3`` is deliberately not hashable here even though `parse_cas_ref`
    still RECOGNISES a blake3 ref: such a ref can only come from a stale
    pointer, and the safe answer to "I cannot check these bytes" is a refusal,
    never a skip.
    """
    if algo == "sha256":
        return sha256_file(path)
    raise ValueError(f"unsupported hash algorithm {algo!r}")


def verify_file_digest(path: Path, ref: str) -> None:
    """Verify a file against an algorithm-tagged ref, DISPATCHING ON THE PREFIX.

    Hardcoding the algorithm per call site is how a sha256 digest gets checked
    with blake3 and every honest file looks corrupt. An empty or undigestable ref
    RAISES out of `parse_cas_ref`; there is no "nothing to check" path here.
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


class _Mirror:
    """A SECOND positional destination written in the same pass.

    The endpoint volume, filled at the same time as local CAS rather than by a
    full re-read afterwards. Best effort in the strongest sense: the first
    error of any kind disables it permanently for this file and is never
    propagated — a volume that is full, slow, read-only or absent must cost the
    request nothing. It is published only by ``close(publish=True)``, after the
    LOCAL file has passed its whole-file digest, and it stages under a
    WRITER-UNIQUE name so a concurrent pod on the same volume can neither
    observe our partial bytes nor have its published blob written into.
    """

    __slots__ = ("fd", "tmp", "dst", "_ok", "_lock")

    def __init__(self, fd: int, tmp: Path, dst: Path) -> None:
        self.fd = fd
        self.tmp = tmp
        self.dst = dst
        self._ok = True
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return self._ok

    def disable(self, why: BaseException) -> None:
        with self._lock:
            if not self._ok:
                return
            self._ok = False
        _log.warning("mirror_disabled destination=%s: %s", self.dst.name, why)

    def write(self, data: bytes, offset: int) -> None:
        if not self._ok:
            return
        try:
            _pwrite_all(self.fd, data, offset)
        except OSError as exc:
            self.disable(exc)

    def close(self, *, publish: bool) -> bool:
        """Publish (or discard) the mirror. Never raises."""
        try:
            if not (publish and self._ok):
                return False
            os.fsync(self.fd)
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            self.tmp.replace(self.dst)
            from .cozy_cas import fsync_dir

            fsync_dir(self.dst.parent)
            return True
        except OSError as exc:
            _log.warning("mirror_publish_failed destination=%s: %s", self.dst, exc)
            return False
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.tmp.unlink(missing_ok=True)


def volume_chunk_dir(chunks_root: Path, ref: str) -> Path:
    """Where one file's verified chunk objects live on the endpoint volume.

    ``chunks/<algo>/aa/bb/<filehex>/`` — the same fanout as ``blobs/``, and a
    SIBLING of it rather than a tenant inside it, so nothing that walks the blob
    tree can mistake a chunk for a blob. Scoping the directory by the FILE
    digest is what makes cleanup a single unambiguous act by a single owner: the
    set of objects belonging to a completed file is exactly one directory, not a
    query over a shared pool.
    """
    algo, hexpart = parse_cas_ref(ref)
    return chunks_root / algo / hexpart[:2] / hexpart[2:4] / hexpart


def _chunk_object_name(index: int, sha256: str) -> str:
    """``<index>-<chunk digest>``, zero-padded so the directory sorts numerically.

    All three coordinates a chunk has are in the name: the file (the directory),
    its position (the index) and its content (the digest). The digest is what
    makes a write IDEMPOTENT — two pods that reach this name wrote the same
    bytes — and the index is what lets an adopter build the name straight from
    the manifest instead of inventorying the directory. A file re-chunked at a
    different size therefore addresses different names and cannot be adopted by
    accident.
    """
    return f"{index:08d}-{sha256}"


def drop_volume_chunks(directory: Path) -> int:
    """Remove one file's chunk objects; the volume now holds the whole blob.

    **Who and when:** whoever ESTABLISHES that the volume holds the complete
    blob under its digest name — the pod that publishes it (tee or copy), or a
    pod that finds it already published. Every such site calls this, so the only
    chunk sets that survive are those of a file no pod has ever completed on
    this volume, and the moment one does they go. That needs no sweep, no owner
    registry and no age clock.

    **Under concurrency:** ``unlink`` is a NAMESPACE operation, so an adopter
    that already opened an object keeps reading its inode to the end, and one
    that opens after the unlink simply misses and refetches that chunk. Neither
    can read WRONG bytes — every adopted chunk is re-hashed regardless. In-flight
    writers' staged files are left alone (they are dot-prefixed and
    writer-unique); the ``rmdir`` then fails and is ignored, and the next
    completion collects the directory.
    """
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if entry.name.startswith("."):
            continue  # another writer's staged chunk, still being written
        try:
            entry.unlink()
            removed += 1
        except OSError:
            pass
    with contextlib.suppress(OSError):
        directory.rmdir()
    if removed:
        _log.info("volume_chunks_dropped path=%s objects=%d", directory.name[:16], removed)
    return removed


class _VolumeChunks:
    """The endpoint volume's verified-chunk store for ONE incomplete file.

    Best effort in exactly ``_Mirror``'s sense: the first error of any kind
    disables the store for this file and is never propagated, because a volume
    that is full, slow, read-only or absent must cost the request nothing.

    The volume is mounted by ONE endpoint, so a bad chunk here is self-inflicted
    rather than hostile — which is why a rejected object is simply unlinked and
    refetched. It is NOT single-writer, though: two pods of that endpoint are
    concurrent, so every write stages writer-unique and renames atomically, and
    every read re-hashes.
    """

    __slots__ = ("dir", "_ok", "_lock")

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self._ok = True
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return self._ok

    def disable(self, why: BaseException) -> None:
        with self._lock:
            if not self._ok:
                return
            self._ok = False
        _log.warning("volume_chunks_disabled path=%s: %s", self.dir.name[:16], why)

    def path_for(self, index: int, sha256: str) -> Path:
        return self.dir / _chunk_object_name(index, sha256)

    def publish(self, index: int, spec: ChunkSpec, fd: int, offset: int) -> bool:
        """Copy one VERIFIED chunk from the local part file onto the volume.

        The read is of bytes this process wrote moments ago, so it comes off the
        page cache; the cost is one volume write, taken while the pool is still
        waiting on the network. Staged writer-unique and renamed, so the final
        name only ever appears with complete bytes — the volume holds verified
        chunks, never a partial one.
        """
        if not self._ok:
            return False
        final = self.path_for(index, spec.sha256)
        tmp = self.dir / f".{final.name}.part-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "xb") as out:
                pos, remaining = offset, spec.length
                while remaining > 0:
                    b = os.pread(fd, min(_READ_CHUNK_BYTES, remaining), pos)
                    if not b:
                        raise OSError(f"short read publishing chunk {index}")
                    out.write(b)
                    pos += len(b)
                    remaining -= len(b)
            # Idempotent by construction: the name fixes the content, so a
            # racing pod's replace puts the SAME bytes there. Atomic, so a
            # concurrent reader sees one complete version or the other.
            os.replace(tmp, final)
            return True
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            self.disable(exc)
            return False

    def adopt(
        self,
        chunks: Sequence[ChunkSpec],
        offsets: Sequence[int],
        wanted: Sequence[int],
        fd: int,
        mirror: Optional["_Mirror"],
        window: int,
    ) -> set[int]:
        """Take the chunks a PREVIOUS POD verified and left on the volume.

        Every adopted chunk is RE-HASHED here before it is trusted: the name is
        a claim, not a proof, so a corrupt or truncated object costs a wasted
        hash pass and can never inject bytes. Rejected objects are unlinked, so
        the next pod does not pay the same pass again.

        Bytes go into the local part file (and the mirror) as they are hashed.
        Writing before the verdict is safe for the same reason a failed fetch
        attempt is: the range belongs to this chunk alone and nothing reads it
        until the chunk is marked done, so a rejected adoption is simply
        overwritten by the refetch.
        """
        if not self._ok or not wanted:
            return set()

        def _take(i: int) -> Optional[int]:
            spec = chunks[i]
            src = self.path_for(i, spec.sha256)
            h = hashlib.sha256()
            pos, remaining = offsets[i], spec.length
            try:
                with open(src, "rb") as f:
                    while remaining > 0:
                        b = f.read(min(_READ_CHUNK_BYTES, remaining))
                        if not b:
                            return None
                        h.update(b)
                        _pwrite_all(fd, b, pos)
                        if mirror is not None:
                            mirror.write(b, pos)
                        pos += len(b)
                        remaining -= len(b)
                    if f.read(1):
                        return None  # longer than the manifest says it is
            except FileNotFoundError:
                return None
            except OSError as exc:
                self.disable(exc)
                return None
            if h.hexdigest() != spec.sha256:
                _log.warning(
                    "volume_chunk_corrupt path=%s index=%d: re-hash rejected it; refetching",
                    self.dir.name[:16], i,
                )
                with contextlib.suppress(OSError):
                    src.unlink()
                return None
            return i

        with ThreadPoolExecutor(max_workers=max(1, min(window, len(wanted)))) as pool:
            kept = {i for i in pool.map(_take, wanted) if i is not None}
        adopted_bytes = sum(chunks[i].length for i in kept)
        _log.info(
            "chunk_resume source=volume path=%s adopted=%d/%d chunks bytes=%d",
            self.dir.name[:16], len(kept), len(wanted), adopted_bytes,
        )
        return kept

    def drop(self) -> int:
        return drop_volume_chunks(self.dir)


def _open_volume_chunks(directory: Optional[Path]) -> Optional["_VolumeChunks"]:
    return None if directory is None else _VolumeChunks(directory)


def _retry_delay(strikes: int) -> float:
    """Full-jitter exponential backoff, capped. Same shape as the whole-file
    loop in ``cozy_cas``."""
    return random.uniform(0.5, 1.0) * min(_CHUNK_BACKOFF_CAP_S, 2.0 ** strikes)


def _fetch_chunk_to_offset(
    session: requests.Session,
    spec: ChunkSpec,
    fd: int,
    offset: int,
    *,
    on_bytes: Optional[Callable[[int], None]],
    mirror: Optional["_Mirror"] = None,
    give_up: Optional[threading.Event] = None,
) -> None:
    """Stream ONE chunk straight into its byte range, verifying as it goes.

    Abandon-and-refetch is per attempt and per chunk: an attempt that stops
    clearing its own progress floor is dropped and retried on a FRESH connection,
    so one bad connection cannot hold the file. That is split from the case where
    NOTHING arrived — see ``_CHUNK_MAX_ATTEMPTS`` above.

    The chunk is verified AFTER it lands rather than before, which is safe
    precisely because the range belongs to this chunk alone: an attempt that
    fails verification is overwritten by the retry, and nothing reads the range
    until the caller marks the chunk done. That is the trade that drops RAM per
    worker from a whole 64 MiB chunk to one read block.
    """
    strikes = 0
    attempt = 0
    while True:
        attempt += 1
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
                    if mirror is not None:
                        mirror.write(block, pos)
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
            # Backoff is for a route that is NOT THERE, and nothing else.
            #   * A DigestMismatch means a complete, wrong-hashing body arrived:
            #     the route is demonstrably healthy, the bytes are not. Retry at
            #     once — sleeping would only delay the refetch of an object the
            #     store is happy to serve.
            #   * A transport error that still CLEARED THE FLOOR is a lemon
            #     connection: get a fresh socket now, not in 30s.
            #   * A transport error that delivered nothing is the case that
            #     needs to wait.
            alive = isinstance(exc, DigestMismatch) or floor.cleared(delivered)
            strikes = 0 if alive else strikes + 1
            out_of_budget = (
                attempt >= _CHUNK_MAX_ATTEMPTS
                or (give_up is not None and give_up.is_set())
            )
            delay = 0.0 if (alive or out_of_budget) else _retry_delay(strikes)
            _log.warning(
                "chunk_refetch sha256=%s attempt=%d/%d delivered=%d "
                "route_alive=%s retry_in=%.3fs: %s",
                spec.sha256[:12], attempt, _CHUNK_MAX_ATTEMPTS,
                delivered, alive, delay, exc,
            )
            if out_of_budget:
                raise
            if delay and give_up is not None:
                # A sibling chunk's fatal failure must not have to wait out
                # this loop's whole budget before the pool can drain.
                if give_up.wait(delay):
                    raise
            elif delay:
                time.sleep(delay)


@contextlib.contextmanager
def _cas_fetch_lock(dst: Path) -> Iterator[None]:
    """An OS ``flock`` on a per-digest lock file so G compute children sharing
    ONE container filesystem dedup a fetch of the same CAS entry to ONE download
    instead of G. An in-process ``threading.Lock`` cannot: the groups are
    separate PROCESSES. The first child to arrive holds the lock and fetches; the
    rest block on ``flock`` and, when it releases, find ``dst`` already present
    and skip the download (the caller's existence check).

    Best-effort: a platform without ``fcntl`` yields without a lock, and the
    writer-unique ``tmp`` path already makes concurrent writers non-corrupting,
    so the lock is a COST optimisation, never a safety gate."""
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
    mirror_dst: Optional[Path] = None,
    mirror_chunk_dir: Optional[Path] = None,
) -> bool:
    """Reassemble a chunked file, verifying every chunk AND the whole-file hash.

    The whole-file hash is fused into the commit stream: there is exactly one
    pass over the bytes, and no second read. On mismatch the partial file is
    deleted so the next attempt starts clean.

    ``mirror_dst`` fills a SECOND store — the endpoint volume — in the same pass,
    so the write-through costs neither a re-read nor a second hash of every byte.
    Returns whether that mirror was published; ``False`` (including always when
    ``mirror_dst`` is ``None``) means the caller still owes the volume a copy.

    ``mirror_chunk_dir`` is that file's chunk-object directory on the same
    volume — see :func:`volume_chunk_dir`. Verified chunks are published
    there as they land and adopted from there on the next pod, so a pod that
    dies 90% into a 35 GB component leaves its successor 3.5 GB to fetch instead
    of 35 GB.
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

    # G compute children share ONE container filesystem, so serialise the fetch
    # of this exact CAS entry across processes (G x egress -> 1x). The lock is a
    # COST optimisation: the writer-unique tmp path already makes concurrent
    # writers non-corrupting, so a lock we cannot take costs a duplicate
    # download, never correctness.
    with _cas_fetch_lock(dst):
        if dst.exists() and dst.stat().st_size == total_size:
            # A sibling in this pod already materialised it while we waited.
            _log.info("chunk_cas dedup: %s already present (%d bytes); skipping "
                      "the fetch", dst.name, total_size)
            return False
        return _download_chunked_locked(
            chunks, dst, want_whole=want_whole, total_size=total_size,
            window=window, on_bytes=on_bytes, session_factory=session_factory,
            mirror_dst=mirror_dst, mirror_chunk_dir=mirror_chunk_dir,
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
    range it claims is re-hashed off disk here, so a write that never reached the
    platter is simply refetched. This is what buys resume WITHOUT an fsync per
    chunk, and it resumes out of order.
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
    just written — from page cache in the normal case — which is the price of not
    serialising every worker behind one in-order commit.
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
    throughput REGRESSION at window 16 and 32."""
    try:
        from requests.adapters import HTTPAdapter

        adapter = HTTPAdapter(pool_connections=max(10, window),
                              pool_maxsize=max(10, window))
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:  # noqa: BLE001 - a caller-supplied session may not mount
        pass


def _open_mirror(mirror_dst: Optional[Path], total_size: int) -> Optional["_Mirror"]:
    """Stage the volume-side part file, or return ``None``. Never raises: a
    volume we cannot write to costs the request nothing but the write-through."""
    if mirror_dst is None:
        return None
    try:
        mirror_dst.parent.mkdir(parents=True, exist_ok=True)
        # Writer-unique, exactly as `_copy_verified_blob` stages: several pods
        # share one volume with no lock between them, and a stable name would
        # let one pod's in-flight bytes land in another's published blob.
        tmp = mirror_dst.parent / (
            f".{mirror_dst.name}.mirror-{os.getpid()}-{uuid.uuid4().hex}"
        )
        fd = os.open(str(tmp), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        _log.warning("mirror_open_failed destination=%s: %s", mirror_dst, exc)
        return None
    try:
        # `ftruncate` ONLY — deliberately NOT `_preallocate`. glibc's
        # `posix_fallocate` zero-fills by hand when the filesystem has no
        # `fallocate(2)`, and network storage frequently does not (NFSv4.1),
        # costing a full extra write pass to the volume. Local CAS keeps the real
        # up-front allocation — "prove the disk now" belongs there — and the
        # volume running out of space just disables the mirror.
        os.ftruncate(fd, total_size)
    except OSError as exc:
        _log.warning("mirror_allocate_failed destination=%s: %s", mirror_dst, exc)
        os.close(fd)
        tmp.unlink(missing_ok=True)
        return None
    return _Mirror(fd, tmp, mirror_dst)


def _seed_mirror(
    fd: int,
    mirror: "_Mirror",
    chunks: Sequence[ChunkSpec],
    offsets: Sequence[int],
    already: set[int],
) -> None:
    """Copy resumed (already-verified) ranges from local disk into the mirror."""
    for i in sorted(already):
        pos, remaining = offsets[i], chunks[i].length
        while remaining > 0 and mirror.ok:
            try:
                b = os.pread(fd, min(_READ_CHUNK_BYTES, remaining), pos)
            except OSError as exc:
                mirror.disable(exc)
                return
            if not b:
                mirror.disable(OSError(f"short read seeding mirror chunk {i}"))
                return
            mirror.write(b, pos)
            pos += len(b)
            remaining -= len(b)


def _download_chunked_locked(
    chunks: Sequence[ChunkSpec],
    dst: Path,
    *,
    want_whole: str,
    total_size: int,
    window: int,
    on_bytes: Optional[Callable[[int], None]],
    session_factory: Callable[[], requests.Session],
    mirror_dst: Optional[Path] = None,
    mirror_chunk_dir: Optional[Path] = None,
) -> bool:
    """The reassembly itself, run while holding the cross-process CAS fetch lock
    (``download_chunked_file`` does the shape checks and the dedup)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The part file name is STABLE, which is what makes resume reachable at all.
    # It is safe because writers of one CAS entry write IDENTICAL bytes to
    # IDENTICAL offsets — positional writes make concurrent assembly of the same
    # digest convergent rather than corrupting.
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
    mirror: Optional[_Mirror] = None
    published = False
    try:
        _preallocate(fd, total_size, tmp)
        already = _adopt_partial(fd, chunks, offsets, journal, header, max_inflight)
        mirror = _open_mirror(mirror_dst, total_size)
        if mirror is not None and already:
            # Resumed ranges are already verified on local disk and will not be
            # refetched, so the mirror takes them from there — otherwise it
            # would publish a file with holes.
            _seed_mirror(fd, mirror, chunks, offsets, already)

        # Whatever THIS pod's local journal could not supply, a PREVIOUS pod may
        # have left verified on the volume. Local disk wins; everything else is
        # adopted here, re-hashed, and never fetched again.
        vol_chunks = _open_volume_chunks(mirror_chunk_dir)
        if vol_chunks is not None:
            already = already | vol_chunks.adopt(
                chunks, offsets,
                [i for i in range(len(chunks)) if i not in already],
                fd, mirror, max_inflight,
            )

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
                mirror=mirror, vol_chunks=vol_chunks,
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
        # ONLY here: the local file just proved its whole-file digest, and the
        # mirror took the same in-memory blocks at the same offsets.
        published = mirror is not None and mirror.close(publish=True)
        mirror = None
        if mirror_dst is not None:
            _log.info("blob_fill_publish source=r2 destination=volume mode=tee "
                      "digest=%s bytes=%d published=%s",
                      want_whole[:16], total_size, published)
        # The volume now holds the COMPLETE blob under its digest name, so this
        # file's chunk objects are garbage. `exists()` rather than `published` —
        # a concurrent pod may have published it first, and the chunks are just
        # as dead either way.
        if vol_chunks is not None and mirror_dst is not None and mirror_dst.exists():
            vol_chunks.drop()
        return published
    finally:
        if mirror is not None:
            mirror.close(publish=False)
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
    mirror: Optional["_Mirror"] = None,
    vol_chunks: Optional["_VolumeChunks"] = None,
) -> None:
    pending = [i for i in range(len(chunks)) if i not in already]
    if not pending:
        return

    aggregate = ProgressFloor(_CHUNK_PROGRESS_FLOOR_BYTES * max_inflight)
    delivered_total = 0
    counter = threading.Lock()
    # Per-chunk retries sleep, so a fatal failure elsewhere must be able to cut
    # them short — otherwise the pool cannot drain until every sibling has spent
    # its whole retry budget.
    give_up = threading.Event()

    def _count(n: int) -> None:
        nonlocal delivered_total
        with counter:
            delivered_total += n
        if on_bytes is not None:
            on_bytes(n)

    session = session_factory()
    _size_session_pool(session, max_inflight)

    def _one(i: int) -> None:
        _fetch_chunk_to_offset(
            session, chunks[i], fd, offsets[i], on_bytes=_count,
            mirror=mirror, give_up=give_up,
        )
        # O_APPEND makes a short line atomic across threads, so the journal
        # needs no lock of its own.
        os.write(jfd, f"{i}\n".encode())
        prefix.mark(i)
        # The chunk has proved its own digest, so it is publishable on its own —
        # AFTER the mark, which keeps the whole-file hasher moving while this
        # worker pays the volume write.
        if vol_chunks is not None:
            vol_chunks.publish(i, chunks[i], fd, offsets[i])

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
                    give_up.set()
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
    # Durable atomic finalize: rename is atomic in the NAMESPACE only.
    # CYCLE: cozy_cas imports DigestMismatch from this module.
    from .cozy_cas import fsync_dir

    os.fsync(fd)
    tmp.replace(dst)
    fsync_dir(dst.parent)
    journal.unlink(missing_ok=True)
    _log.info("chunk_download_done path=%s size=%d sha256=%s", dst.name, total_size, got[:16])
