"""Chunked sha256 CAS upload client (pgw#781 / th#1303).

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
*   **Resume by RE-DECLARING.** There is no session state to reconstruct: the
    need-set simply comes back smaller, because the chunks that landed are now
    resident. A kill mid-upload loses at most the in-flight chunks.
*   **Parallel across chunks**, bounded, sharing the process-wide PUT budget so
    file-level and chunk-level fan-out cannot multiply into a retry storm.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import requests

from .chunk_cas import CAS_CHUNK_SIZE_BYTES, chunk_count_for, chunk_len_at

__all__ = [
    "ChunkPlan",
    "FileDeclaration",
    "UploadGrant",
    "UploadReport",
    "hash_file_and_chunks",
    "upload_grants",
]

_log = logging.getLogger(__name__)

_READ_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_ATTEMPTS = 5

# Shared with the legacy presigned path's intent: total concurrent PUTs across
# every file and chunk. Two fan-out axes that each look modest multiply into a
# retry storm otherwise.
_PUT_BUDGET = 8
_put_slots = threading.BoundedSemaphore(_PUT_BUDGET)


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


@dataclass(frozen=True)
class UploadGrant:
    """One `need` entry from the hub: where to PUT, and with which headers."""

    digest: str  # "sha256:<hex>" — also the CAS key
    size_bytes: int
    put_url: str
    headers: Dict[str, str]
    staging_key: str = ""


@dataclass
class UploadReport:
    """Outcome plus the DENOMINATORS. A caller must be able to tell a full
    dedup hit from a run that uploaded nothing because it examined nothing."""

    granted: int = 0
    uploaded: int = 0
    bytes_uploaded: int = 0
    skipped_resident: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and self.uploaded == self.granted


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


def _put_one(
    session: requests.Session,
    grant: UploadGrant,
    body: bytes,
) -> None:
    """One single PUT, headers VERBATIM.

    A 4xx that is not 408/429 is terminal: with the checksum inside the
    signature, 400 means our bytes disagree with the digest we computed (a
    local bug or a file that changed under us) and 403 means the grant does not
    authorize what we sent. Neither is fixed by retrying.
    """
    last = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with _put_slot():
                resp = session.put(
                    grant.put_url,
                    data=body,
                    headers=dict(grant.headers),
                    timeout=(60, 300),
                )
            code = int(resp.status_code)
            if 200 <= code < 300:
                return
            body_sample = (resp.text or "")[:300]
            if 400 <= code < 500 and code not in (408, 429):
                raise ValueError(
                    f"chunk {grant.digest[:20]}… refused with HTTP {code}: {body_sample}"
                )
            last = f"HTTP {code}: {body_sample}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "chunk_put_retry digest=%s attempt=%d/%d: %s",
            grant.digest[:20], attempt, _MAX_ATTEMPTS, last,
        )
    raise ValueError(f"chunk {grant.digest[:20]}… failed after {_MAX_ATTEMPTS} attempts: {last}")


class _put_slot:
    def __enter__(self) -> "_put_slot":
        _put_slots.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        _put_slots.release()


def upload_grants(
    grants: Sequence[UploadGrant],
    source_for: Callable[[str], "tuple[Path, int, int]"],
    *,
    parallel: int = 4,
    session_factory: Callable[[], requests.Session] = requests.Session,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> UploadReport:
    """PUT every granted object.

    ``source_for(digest)`` returns ``(path, offset, length)`` — where the bytes
    for that CAS object live locally. A whole file is ``(path, 0, size)``; a
    chunk is its span. Only GRANTED objects are uploaded: anything the hub
    reported as `have` was never granted and is silently, correctly, skipped.
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
            body = _read_span(path, offset, length)
            # Verify our own bytes before spending the transfer. The store
            # would refuse them anyway, but a local mismatch is a local bug and
            # the error should say so rather than arriving as an opaque 400.
            got = hashlib.sha256(body).hexdigest()
            want = g.digest.split(":", 1)[-1]
            if got != want:
                raise ValueError(
                    f"local bytes for {g.digest[:20]}… hash to {got[:12]}… — refusing to upload"
                )
            _put_one(session, g, body)
            with lock:
                rep.uploaded += 1
                rep.bytes_uploaded += length
            if on_bytes is not None:
                on_bytes(length)
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
        "chunk_upload granted=%d uploaded=%d bytes=%d failed=%d",
        rep.granted, rep.uploaded, rep.bytes_uploaded, len(rep.failures),
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
