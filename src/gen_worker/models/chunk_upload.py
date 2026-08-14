"""Chunked sha256 CAS upload client.

The publisher's side of the v2 flow. Shape, and why each part is the way it is:

*   **ONE streaming pass** computes the whole-file sha256 AND every per-chunk
    sha256 (same read, two hashers, boundaries at fixed 64 MiB offsets). A
    second pass over multi-GB shards is the thing this design exists to avoid.
*   **Files ≤ 64 MiB stay WHOLE** — one object, and its sha256 is
    simultaneously its CAS key. Only files above the threshold get a chunk
    list, so configs/tokenizers/index json do not multiply into chunk objects.
*   **DECLARE BEFORE UPLOAD.** The hub answers `{have, need}` from store
    residency, so a chunk already in the CAS costs neither an upload nor a
    copy. Nothing is uploaded on the strength of a local belief.
*   **One single PUT per chunk**, carrying the hub's grant headers **VERBATIM**.
    The checksum is a SIGNED HEADER in those grants: R2 then refuses wrong
    bytes (400 BadDigest, and the object does not exist afterwards) and refuses
    a substituted claim (403 SignatureDoesNotMatch).

    This is the one place a "helpful" refactor is genuinely dangerous. The AWS
    SDK's presigner HOISTS the checksum into the query string instead, and
    MEASURED against R2 that form is IGNORED — an honest PUT succeeds with no
    stored checksum at all, which means wrong bytes would also have been
    accepted. So: send exactly the headers the hub gave, add nothing, drop
    nothing, and never reconstruct a URL locally.
*   **Resume by RE-PLANNING.** There is no session state to reconstruct: the
    need-set comes back smaller from ``POST .../grants``, because what landed
    is now accounted for. A kill mid-upload loses at most the in-flight
    objects. That re-plan is also the CAS path's presign re-mint: a grant that
    outlived ``expires_at`` is not fatal, it is re-planned.
*   **Parallel across chunks**, bounded, sharing the process-wide PUT budget so
    file-level and chunk-level fan-out cannot multiply into a retry storm. The
    budget is taken BEFORE the span is read, so it bounds buffer residency and
    not merely socket count.
*   **Retry hygiene identical to the presigned path**:
    decorrelated-jitter backoff from the one shared helper, classify before
    charging an attempt, and a liveness beat per completed object so a healthy
    multi-hour transfer is not silence to the watchdogs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import requests

from .. import activity as _activity
from .. import progress as _progress
from ..hubio.transport import backoff_sleep_s
from ..hubio.transport import GrantExpired as GrantExpired
from ..hubio.transport import PUT_EXPIRED, PUT_OK, PUT_TERMINAL, put_verdict
from .chunk_cas import (
    CAS_CHUNK_SIZE_BYTES,
    _READ_CHUNK_BYTES,
    chunk_count_for,
    chunk_len_at,
)

__all__ = [
    "ChunkPlan",
    "FileDeclaration",
    "GrantExpired",
    "UploadGrant",
    "UploadReport",
    "hash_file_and_chunks",
    "upload_grants",
]

_log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 5

# Total concurrent PUTs across every file and chunk. Two fan-out axes that each
# look modest multiply into a retry storm otherwise.
#
# Derivation, measured rather than guessed: the hub's ranged reader needs
# SIXTEEN windows in flight to move 3.5 MB/s where one stream moves 0.10 MB/s on
# the same class of link (`internal/s3/resumable_reader.go:45`), and the SDK
# transfer path next door runs at 10 workers. 8 in flight per publish under a
# 16-PUT process ceiling sits between the two: it cannot leave a pod uplink
# idle, and it still cannot multiply with a second publish into something the
# store reads as a flood. Chunks are 64 MiB, so the slot is also the buffer
# bound — 8 × 64 MiB = 512 MiB per publish.
_DEFAULT_PARALLEL = 8
_PUT_BUDGET = 16
_put_slots = threading.BoundedSemaphore(_PUT_BUDGET)

# Don't START a PUT on a grant with less than this left: a 64 MiB body over a
# slow link outlives a few seconds of TTL, and a presign that dies mid-flight
# costs a whole classification pass to discover.
_GRANT_EXPIRY_MARGIN_S = 120.0


@dataclass(frozen=True)
class ChunkPlan:
    """One chunk of a file: its digest, its offset and its length."""

    sha256: str
    offset: int
    length: int


@dataclass
class FileDeclaration:
    """What the publisher declares for one file.

    ``chunks`` is empty for a whole file (size ≤ CAS_CHUNK_SIZE_BYTES); the
    hub's manifest schema expects exactly that.
    """

    path: str
    size_bytes: int
    digest: str  # "sha256:<hex>", the WHOLE-file hash
    chunks: List[ChunkPlan] = field(default_factory=list)

    def to_wire(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
        }
        if self.chunks:
            out["chunks"] = [
                {"digest": c.sha256, "len": c.length} for c in self.chunks
            ]
        return out




def _parse_expires_at(raw: str) -> float:
    """RFC3339 -> unix seconds. 0.0 means "the hub named no expiry"."""
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        # `fromisoformat` handles the offset forms; Go's RFC3339 "Z" needs the
        # swap on Python < 3.11 and is harmless on newer ones.
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.timestamp()


@dataclass(frozen=True)
class UploadGrant:
    """One `need` entry from the hub: where to PUT, and with which headers.

    ``expires_at`` is the hub's ``ObjectGrant.ExpiresAt`` (RFC3339). Reading it
    is what makes a 403 attributable — an expired presign re-plans, a
    substituted claim does not. "" means the hub named none, and then no expiry
    check is made.
    """

    digest: str  # "sha256:<hex>" — also the CAS key
    size_bytes: int
    put_url: str
    headers: Dict[str, str]
    staging_key: str = ""
    expires_at: str = ""

    def expires_at_unix(self) -> float:
        return _parse_expires_at(self.expires_at)

    def expired(
        self, *, margin_s: float = _GRANT_EXPIRY_MARGIN_S, now: Optional[float] = None
    ) -> bool:
        """Is this grant past its expiry, or too close to it to start a PUT?"""
        deadline = self.expires_at_unix()
        if deadline <= 0.0:
            return False
        return (now if now is not None else time.time()) + margin_s >= deadline


@dataclass
class UploadReport:
    """Outcome plus the DENOMINATORS. A caller must be able to tell a full
    dedup hit from a run that uploaded nothing because it examined nothing."""

    granted: int = 0
    uploaded: int = 0
    bytes_uploaded: int = 0
    skipped_resident: int = 0
    failures: List[str] = field(default_factory=list)
    #: Digests whose GRANT expired. Re-plannable, never a failure of the bytes.
    expired: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.expired and self.uploaded == self.granted

    @property
    def needs_replan(self) -> bool:
        """Nothing is wrong except stale presigns — a re-plan alone fixes it."""
        return bool(self.expired) and not self.failures


def hash_file_and_chunks(
    path: Path,
    *,
    chunk_size: int = CAS_CHUNK_SIZE_BYTES,
    rel_path: str = "",
) -> FileDeclaration:
    """ONE streaming pass producing the whole-file sha256 AND the chunk list.

    Chunk boundaries are fixed multiples of ``chunk_size``, so a chunk's offset
    is pure arithmetic and matches what the hub validates. A file at or below
    the threshold gets NO chunk list — it is stored whole and its digest is its
    CAS key.
    """
    size = path.stat().st_size
    whole = hashlib.sha256()
    chunks: List[ChunkPlan] = []
    current = hashlib.sha256()
    current_len = 0
    offset = 0

    with open(path, "rb") as f:
        while True:
            want = _READ_CHUNK_BYTES
            if size > chunk_size:
                # Never read across a chunk boundary in one go, or the chunk
                # hasher would absorb bytes belonging to the next chunk.
                want = min(want, chunk_size - current_len)
            block = f.read(want)
            if not block:
                break
            whole.update(block)
            if size > chunk_size:
                current.update(block)
                current_len += len(block)
                if current_len == chunk_size:
                    chunks.append(
                        ChunkPlan(sha256=current.hexdigest(), offset=offset, length=current_len)
                    )
                    offset += current_len
                    current = hashlib.sha256()
                    current_len = 0
    if size > chunk_size and current_len:
        chunks.append(ChunkPlan(sha256=current.hexdigest(), offset=offset, length=current_len))

    decl = FileDeclaration(
        path=rel_path or path.name,
        size_bytes=size,
        digest="sha256:" + whole.hexdigest(),
        chunks=chunks,
    )
    _assert_consistent(decl, chunk_size)
    return decl


def _assert_consistent(decl: FileDeclaration, chunk_size: int) -> None:
    """Check our OWN output against the hub's invariants before declaring it.

    A publisher that ships an inconsistent declaration gets refused at the
    hub — better to fail here, where the file is in hand and the error names
    the local bug.
    """
    want_n = chunk_count_for(decl.size_bytes, chunk_size)
    if want_n != len(decl.chunks):
        raise ValueError(
            f"{decl.path}: produced {len(decl.chunks)} chunks, size {decl.size_bytes} "
            f"at chunk size {chunk_size} needs {want_n}"
        )
    if not decl.chunks:
        return
    total = 0
    for i, c in enumerate(decl.chunks):
        want_len = chunk_len_at(decl.size_bytes, i, chunk_size)
        if c.length != want_len:
            raise ValueError(f"{decl.path}: chunk {i} is {c.length} bytes, needs {want_len}")
        if c.offset != i * chunk_size:
            raise ValueError(f"{decl.path}: chunk {i} offset {c.offset} is not {i * chunk_size}")
        total += c.length
    if total != decl.size_bytes:
        raise ValueError(f"{decl.path}: chunk lengths sum to {total}, size is {decl.size_bytes}")


def _read_span(path: Path, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(length)
    if len(buf) != length:
        raise ValueError(f"{path}: short read at offset {offset} ({len(buf)} of {length})")
    return buf


def _classify_put(grant: UploadGrant, code: int, body_sample: str) -> Optional[Exception]:
    """None = success. Otherwise the typed outcome for this HTTP status.

    With the checksum inside the signature, 400 means our bytes disagree with
    the digest we computed (a local bug, or a file that changed under us) and
    403 normally means the grant does not authorize what we sent. Neither is
    fixed by retrying — EXCEPT that S3 also answers 403 for a presign that has
    simply expired. A 403 on a grant already past its stated ``expires_at`` is
    a re-plan, not a repudiation.
    """
    verdict = put_verdict(code, presign_expired=grant.expired(margin_s=0.0))
    if verdict == PUT_OK:
        return None
    if verdict == PUT_EXPIRED:
        return GrantExpired(
            f"grant for {grant.digest[:20]}… expired at {grant.expires_at} "
            f"(HTTP 403) — re-planning"
        )
    if verdict == PUT_TERMINAL:
        return ValueError(
            f"chunk {grant.digest[:20]}… refused with HTTP {code}: {body_sample}"
        )
    return _Transient(f"HTTP {code}: {body_sample}")


class _Transient(Exception):
    """Retryable in place: 429, 5xx, or a transport-level failure."""


def _put_one(
    session: requests.Session,
    grant: UploadGrant,
    body: bytes,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """One object, headers VERBATIM, with the repo's one backoff policy.

    CLASSIFY BEFORE CHARGING: a terminal status raises on the spot and never
    spends an attempt; only a transient one does, and it pays a
    decorrelated-jitter sleep first (``hubio.transport.backoff_sleep_s`` — the
    same helper the presigned path uses). Never retry without the sleep: that
    is several threads hammering a store that has just answered 429.
    """
    last: Exception = _Transient("no attempt made")
    for attempt in range(1, max_attempts + 1):
        if grant.expired():
            raise GrantExpired(
                f"grant for {grant.digest[:20]}… expires at {grant.expires_at}; "
                f"refusing to start a PUT inside the margin — re-planning"
            )
        try:
            resp = session.put(
                grant.put_url,
                data=body,
                headers=dict(grant.headers),
                timeout=(60, 300),
            )
        except requests.RequestException as exc:
            last = _Transient(f"{type(exc).__name__}: {exc}")
        else:
            outcome = _classify_put(
                grant, int(resp.status_code), (resp.text or "")[:300])
            if outcome is None:
                return
            if not isinstance(outcome, _Transient):
                raise outcome
            last = outcome
        if attempt >= max_attempts:
            break
        delay = backoff_sleep_s(attempt)
        _log.warning(
            "chunk_put_retry digest=%s attempt=%d/%d sleep_s=%.2f: %s",
            grant.digest[:20], attempt, max_attempts, delay, last,
        )
        # Backing off IS work in progress, and the pod must say so: the hub's
        # activity stall window does not care why we are quiet.
        _activity.note_progress()
        sleep(delay)
    raise ValueError(
        f"chunk {grant.digest[:20]}… failed after {max_attempts} attempts: {last}")


class _put_slot:
    def __enter__(self) -> "_put_slot":
        _put_slots.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        _put_slots.release()


def _beat(n: int) -> None:
    """Feed the data plane's liveness the way the presigned path does
    (``presigned_upload.py``'s ``_feed``): bytes onto the open activity's
    counter so the 10 s beat carries a moving number, plus proof-of-life.
    Without it a healthy multi-GB publish and a wedged one are the same silence
    to the orchestrator's activity stall window.
    """
    act = _activity.current()
    if act is not None:
        try:
            act.counter("upload:bytes", _progress.UNIT_BYTES).add(n)
        except Exception:  # noqa: BLE001 — reporting never fails a transfer
            _log.debug("chunk upload beat dropped", exc_info=True)
    _activity.note_progress()


def upload_grants(
    grants: Sequence[UploadGrant],
    source_for: Callable[[str], "tuple[Path, int, int]"],
    *,
    parallel: int = _DEFAULT_PARALLEL,
    session_factory: Callable[[], requests.Session] = requests.Session,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> UploadReport:
    """PUT every granted object.

    ``source_for(digest)`` returns ``(path, offset, length)`` — where the bytes
    for that CAS object live locally. A whole file is ``(path, 0, size)``; a
    chunk is its span. Only GRANTED objects are uploaded: anything the hub
    reported as `have` was never granted and is silently, correctly, skipped.

    A grant that has expired (or expires within the margin) lands in
    ``report.expired`` rather than ``report.failures``: the caller re-plans,
    which re-mints it. See :class:`GrantExpired`.
    """
    rep = UploadReport(granted=len(grants))
    if not grants:
        return rep
    lock = threading.Lock()
    session = session_factory()

    def _one(g: UploadGrant) -> None:
        try:
            path, offset, length = source_for(g.digest)
            if length != g.size_bytes:
                raise ValueError(
                    f"grant for {g.digest[:20]}… is {g.size_bytes} bytes, local span is {length}"
                )
            # The slot is taken BEFORE the span is read, so the PUT budget
            # bounds bytes resident in this process and not merely sockets open
            # from it.
            with _put_slot():
                body = _read_span(path, offset, length)
                # Verify our own bytes before spending the transfer. The store
                # would refuse them anyway, but a local mismatch is a local bug
                # and the error should say so rather than arriving as an opaque
                # 400.
                got = hashlib.sha256(body).hexdigest()
                want = g.digest.split(":", 1)[-1]
                if got != want:
                    raise ValueError(
                        f"local bytes for {g.digest[:20]}… hash to {got[:12]}… "
                        f"— refusing to upload"
                    )
                _put_one(session, g, body)
            with lock:
                rep.uploaded += 1
                rep.bytes_uploaded += length
            _beat(length)
            if on_bytes is not None:
                on_bytes(length)
        except GrantExpired as exc:
            _log.info("chunk_grant_expired digest=%s: %s", g.digest[:20], exc)
            with lock:
                rep.expired.append(g.digest)
        except (ValueError, OSError) as exc:
            with lock:
                rep.failures.append(f"{g.digest}: {exc}")

    try:
        workers = max(1, min(parallel, len(grants)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, grants))
    finally:
        session.close()
    _log.info(
        "chunk_upload granted=%d uploaded=%d bytes=%d failed=%d expired=%d",
        rep.granted, rep.uploaded, rep.bytes_uploaded,
        len(rep.failures), len(rep.expired),
    )
    return rep


def sources_from_declarations(
    decls: Sequence[FileDeclaration], paths: Dict[str, Path]
) -> Dict[str, "tuple[Path, int, int]"]:
    """Build the digest → (path, offset, length) index the uploader needs.

    Shared bytes resolve once: the same chunk appearing in two files, or twice
    in one file, is uploaded a single time.
    """
    out: Dict[str, tuple] = {}
    for d in decls:
        p = paths[d.path]
        if not d.chunks:
            out.setdefault(d.digest, (p, 0, d.size_bytes))
            continue
        for c in d.chunks:
            out.setdefault("sha256:" + c.sha256, (p, c.offset, c.length))
    return out
