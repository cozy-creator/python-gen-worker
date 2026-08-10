from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from .chunk_cas import (
    CAS_CHUNK_SIZE_BYTES,
    ChunkSpec,
    download_chunked_file,
    drop_volume_chunks,
    hash_file,
    hasher_for,
    parse_cas_ref,
    verify_file_digest,
    volume_chunk_dir,
)
from .cozy_cas import _download_one_file as _download_one_file
from .cozy_cas import _norm_rel_path, fsync_dir, fsync_file
from .download import components_present, select_component_paths
from .errors import PickleWeightRefused
from .hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from .refs import TensorhubRef
from .. import activity as _activity
from .. import boot_phases
from ..capability import InsufficientDiskError
from ..s3_transfer import S3TransferGrant, download_file_with_grant
from .loading import safetensors_file_valid

_log = logging.getLogger("gen_worker.download")

# Per-digest inflight downloads (gw#479 multi-lane fallout, th#757 forensics):
# two refs materializing concurrently SHARE blobs in the content-addressed
# store (split-vendor lanes: 9.7GB of identical encoder shards). Without a
# per-digest lock both tasks streamed into the SAME .part file, interleaved
# writes failed size/blake3 verification 3x, and the second ref died with
# download_failed on every attempt (J24M runs 10-12, three pods, ~2.5min in —
# while every blob verified byte-perfect in R2). First task downloads; the
# second awaits the lock, re-checks usability, and reuses the finished blob.
_INFLIGHT_BLOB_LOCKS: dict[str, "asyncio.Lock"] = {}
_INFLIGHT_BLOB_LOCKS_CAP = 8192


def _inflight_blob_lock(digest: str) -> "asyncio.Lock":
    if len(_INFLIGHT_BLOB_LOCKS) > _INFLIGHT_BLOB_LOCKS_CAP:
        _INFLIGHT_BLOB_LOCKS.clear()  # bounded memory; races just re-lock
    return _INFLIGHT_BLOB_LOCKS.setdefault(digest, asyncio.Lock())


# gw#598 / th#850: verify-on-first-use-per-process for REUSED CAS state.
# The CAS root persists across pod restarts (cozy-local) and, historically,
# could itself be a volume shared by several pods (superseded by gw#599's
# managed-tier ruling — the root is local-only now, but an operator can
# still point TENSORHUB_CACHE_DIR at a shared path manually). Either way,
# blobs and materialized snapshot trees found on disk may be bytes from a
# different process/era — they must pass a full BLAKE3 check once per
# process before being trusted, exactly like freshly downloaded bytes. Keyed
# by (resolved root, digest/key) so tests with distinct tmp roots never
# share trust; in production the root is constant, so this is once per boot.
_TRUST_CAP = 65536
_VERIFIED_BLOBS: Set[tuple] = set()
_TRUSTED_SNAPSHOTS: Set[tuple] = set()


def _mark_trusted(s: Set[tuple], item: tuple) -> None:
    if len(s) > _TRUST_CAP:
        s.clear()  # bounded memory; worst case is a re-verify
    s.add(item)

ProgressFn = Callable[[int, Optional[int]], None]

# th#850 managed-tier ruling (gw#599): bytes actually fetched over the
# network (R2 origin), as opposed to bytes served from a warm local/volume
# cache — the signal a "volume-attached boot ⇒ ~0 network bytes" runtime
# assertion needs. Carried via a ContextVar (same idiom as
# provision.py's ArmingScope) scoped to one ensure_local/ensure_snapshot
# call, so no download/stub call site anywhere needs a new parameter to
# forward it — a caller that never opens the scope sees no behavior change.
_NETWORK_BYTES_SINK: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "cozy_snapshot_network_bytes_sink", default=None
)


class NetworkBytesScope:
    """Capture network-fetched bytes for one download call.

    ``with NetworkBytesScope() as scope: path = await ensure_local(...)``
    then ``scope.network_bytes`` is the total bytes this call fetched from
    R2 (0 if every blob was already warm locally or on the volume).
    """

    def __init__(self) -> None:
        self._sink: List[int] = [0]
        self._token: Optional["contextvars.Token"] = None

    def __enter__(self) -> "NetworkBytesScope":
        self._token = _NETWORK_BYTES_SINK.set(self._sink)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _NETWORK_BYTES_SINK.reset(self._token)
            self._token = None

    @property
    def network_bytes(self) -> int:
        return self._sink[0]

# Free space that must remain after downloading the missing blobs.
_DISK_HEADROOM_BYTES = 1 << 30


# ---------------------------------------------------------------------------
# Snapshot build coordination (threading-based, works across event loops)
# ---------------------------------------------------------------------------

class _SnapshotEntry:
    """One builder, zero-or-more waiters."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.exception: Optional[BaseException] = None


_SNAP_LOCK = threading.Lock()
_SNAP_ENTRIES: Dict[str, _SnapshotEntry] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blob_path(blobs_root: Path, ref: str) -> Path:
    """Local CAS path for an algorithm-tagged ref: ``blobs/<algo>/aa/bb/<hex>``.

    The algorithm is a PATH SEGMENT, mirroring the hub's `blobs/<algo>/` layout,
    so a sha256 blob and a blake3 blob of different bytes can never collide on
    one name and a legacy blake3 tree keeps its exact existing paths. An
    untagged ref is REFUSED by `parse_cas_ref` (pgw#871/th#1357) — a bare hex
    names no namespace, so there is no path to build.
    """
    algo, hexpart = parse_cas_ref(ref)
    return blobs_root / algo / hexpart[:2] / hexpart[2:4] / hexpart


def _component_of(path: str) -> str:
    """The diffusers COMPONENT a snapshot file belongs to (pgw#1087).

    The first path segment when the file sits in a subdirectory, ``(root)``
    otherwise — model_index.json, a top-level scheduler config and the like.
    Deliberately NOT validated against `component_vocab`: an unrecognized
    directory is still a real slice of the download, and dropping it into an
    "other" bucket would make the component rows stop summing to the fetch.
    """
    rel = _norm_rel_path(str(path or ""))
    head, sep, _ = rel.partition("/")
    return head if sep and head else "(root)"


#: pgw#1087: the no-op source sink used when component spans are off (steady
#: state). Named rather than an inline lambda so `_dl` has one shape.
def _NO_SOURCE(_source: str) -> None:
    return None


_PART_FILE_RE = re.compile(r"\.part\d{4}$")


def _is_part_file(path: str) -> bool:
    return bool(_PART_FILE_RE.search(path))


def _is_parts_manifest(path: str) -> bool:
    return path.endswith(".parts.json")


def _copy_verified_blob(src: Path, dst: Path, ref: str, expected_size: int) -> bool:
    """Copy one immutable CAS blob through a writer-unique atomic stage,
    verifying declared size and the CONTENT DIGEST before publishing (th#850
    managed-tier ruling, gw#599). Used both to fill local CAS from the volume
    fill source and to write a fresh R2 fetch through to the volume. The final
    path is digest-only; readers never observe partial bytes, and racing
    writers may replace the same final name only after each independently
    verifies size and digest (mirrors the multi-writer discipline gw#597
    established for ordinary downloads).

    ``ref`` is ALGORITHM-TAGGED and the hash dispatches on it (th#1303): a
    sha256 blob checked with blake3 fails every honest copy, and an empty
    digest must never reduce this to a size-only check.

    pgw#971: the hash is FUSED INTO THE COPY, per pgw#769 — hashing stays
    mandatory, but it rides the read this function already performs instead of
    costing a second full pass. The old shape was `copyfileobj` and then
    `hash_file(tmp)`: three passes over every byte (read source, write stage,
    RE-READ the stage), and on the volume->local direction the re-read was of
    the freshly-written NETWORK-storage file, the single most expensive pass of
    the three. One pass now, same guarantee: the bytes that reach the stage are
    exactly the bytes that went through the hasher.
    """
    algo, want_hex = parse_cas_ref(ref)
    digest = want_hex
    try:
        if not src.is_file():
            return False
        if expected_size and src.stat().st_size != expected_size:
            _log.warning(
                "blob_fill_corrupt source=%s digest=%s reason=size expected=%d actual=%d",
                src, digest[:16], expected_size, src.stat().st_size,
            )
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / (
            f".{dst.name}.writer-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        )
        try:
            hasher = hasher_for(algo)
            with src.open("rb") as source, tmp.open("xb") as staged:
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    hasher.update(block)
                    staged.write(block)
                staged.flush()
                os.fsync(staged.fileno())
            if expected_size and tmp.stat().st_size != expected_size:
                _log.warning(
                    "blob_fill_corrupt source=%s digest=%s reason=staged_size expected=%d actual=%d",
                    src, digest[:16], expected_size, tmp.stat().st_size,
                )
                return False
            if hasher.hexdigest().lower() != want_hex:
                _log.warning(
                    "blob_fill_corrupt source=%s digest=%s reason=%s", src, digest[:16], algo,
                )
                return False
            os.replace(tmp, dst)
            fsync_file(dst)
            fsync_dir(dst.parent)
            return True
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as exc:
        _log.warning(
            "blob_fill_copy_failed source=%s destination=%s digest=%s error=%s",
            src, dst, digest[:16], exc,
        )
        return False


def _try_hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except Exception as e:
        _log.debug("hardlink failed for %s -> %s: %s, trying symlink", src, dst, e)
    try:
        os.symlink(src, dst)
        return
    except Exception as e:
        _log.warning("symlink failed for %s -> %s: %s, falling back to copy", src, dst, e)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Validate the typed resolved manifest (gw#497: WorkerResolvedRepo is THE
# shape — each wire boundary parses into it; no dict-or-object duck typing).
# ---------------------------------------------------------------------------

# pgw#782 / th#1313: pickle serialisation formats a worker REFUSES to
# materialize, mirroring tensorhub's publish-time ban
# (catalog.PickleWeightExtensions). Defence in depth, and the depth is the
# point: the hub now refuses these at publish, but blobs already in the shared
# CAS predate that refusal, and a worker is where unpickling would actually
# execute — `torch.load` / `from_pretrained` on a hostile `.bin` runs the
# attacker's code inside a pod holding our hub credentials and other tenants'
# work. Refusing at RESOLVE means the bytes are never even downloaded.
#
# Paul, 2026-07-30: "definitely ban all of these."
PICKLE_WEIGHT_EXTENSIONS = (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")


def first_pickle_weight_path(paths: Iterable[str]) -> str:
    """First path naming a pickle serialisation format, or "" if none."""
    for raw in paths:
        base = (raw or "").strip().lower().rsplit("/", 1)[-1]
        if base.endswith(PICKLE_WEIGHT_EXTENSIONS):
            return raw
    return ""


def _validate_resolved(ref: TensorhubRef, resolved: WorkerResolvedRepo) -> WorkerResolvedRepo:
    """Normalize digest prefixes and reject unusable manifests."""
    snapshot_digest = (resolved.snapshot_digest or "").strip()
    if not snapshot_digest:
        raise ValueError("resolved model missing snapshot_digest")

    files: List[WorkerResolvedRepoFile] = []
    for f in resolved.files:
        path = (f.path or "").strip()
        if not path:
            continue
        # th#1303: the algorithm-tagged ref FIRST. `cas_ref()` raises on an
        # entry that carries no readable digest, so an unreadable manifest can
        # never degrade into "download it unverified".
        try:
            ref_tagged = f.cas_ref()
        except ValueError as exc:
            raise ValueError(f"resolved model file {path}: {exc}") from exc
        parse_cas_ref(ref_tagged)  # shape-check; the tag travels with the ref
        url = (f.url or "").strip() or None
        transfer_grant = f.transfer_grant if isinstance(f.transfer_grant, dict) else None
        chunks = tuple(f.chunks or ())
        # A chunked entry's bytes exist ONLY as chunks — it legitimately has no
        # whole-file url. Requiring one is how every v2 snapshot would be
        # rejected as untransferable.
        if not chunks and not url and transfer_grant is None:
            raise ValueError(f"resolved model file missing transfer: {path}")
        if chunks:
            declared = sum(int(c.length) for c in chunks)
            size = int(f.size_bytes or 0)
            if size and declared != size:
                raise ValueError(
                    f"resolved model file {path}: chunk lengths sum to {declared}, "
                    f"manifest says {size}"
                )
        files.append(
            WorkerResolvedRepoFile(
                path=path,
                size_bytes=int(f.size_bytes or 0),
                url=url,
                transfer_grant=transfer_grant,
                digest=ref_tagged,
                chunks=chunks,
                chunk_size_bytes=int(f.chunk_size_bytes or 0),
            )
        )

    if not files:
        raise ValueError("resolved model has empty files list")

    if bad := first_pickle_weight_path(f.path for f in files):
        raise PickleWeightRefused(
            f"refusing snapshot {snapshot_digest[:16]} of {ref.canonical()}: "
            f"{bad!r} is a pickle-format weight. Unpickling executes arbitrary "
            f"code in this process; republish the repo with safetensors."
        )

    return WorkerResolvedRepo(snapshot_digest=snapshot_digest, files=files)


#: Directory-key narrowing markers. Written by :func:`snapshot_dir_key` and
#: READ BACK by anyone handed a materialized path who must know whether that
#: path is the whole composition: ``__c`` is a component-scoped subset (the
#: declared components plus root config — loadable on its own), ``__x`` is an
#: override EXCLUSION (th#1330 B2) and is NOT — the excluded component's
#: subfolder is absent by construction, so the tree only loads together with
#: the override trees it was narrowed for (pgw#816).
SUBSET_MARKER = "__c"
EXCLUDE_MARKER = "__x"


def dir_key_excludes_components(path: str | Path) -> bool:
    """Whether a materialized snapshot path is an override-narrowed tree."""
    return EXCLUDE_MARKER in Path(str(path)).name


def snapshot_dir_key(
    snapshot_digest: str,
    components: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> str:
    """On-disk snapshot-directory key (pgw#505): the bare digest normally, or
    ``<digest>__c<fingerprint>`` when ``components`` narrows the materialized
    tree to a SUBSET of the digest's full content. Keyed separately so a
    component-scoped (partial) materialization can never be mistaken for —
    or collide with — the full one under the same digest.

    ``exclude`` (th#1330 B2) narrows the same way from the other side and so
    earns the same treatment: ``__x<fingerprint>``. The bare digest name is
    reserved for a COMPLETE tree, full stop — that reservation is what lets
    the executor's cached-path short-circuit keep trusting a bare-digest
    directory without re-deriving anyone's fetch scope."""
    comps = sorted({c.strip() for c in components if c and str(c).strip()})
    drop = sorted({c.strip() for c in exclude if c and str(c).strip()})
    key = snapshot_digest
    if comps:
        key += SUBSET_MARKER + hashlib.sha1(
            "+".join(comps).encode()).hexdigest()[:12]
    if drop:
        key += EXCLUDE_MARKER + hashlib.sha1(
            "+".join(drop).encode()).hexdigest()[:12]
    return key


def _filter_resolved_components(
    ref: TensorhubRef,
    res: WorkerResolvedRepo,
    components: Sequence[str],
    exclude: Sequence[str] = (),
) -> WorkerResolvedRepo:
    """Narrow a validated :class:`WorkerResolvedRepo` to the declared
    pipeline components (+ root config files) — the tensorhub-source twin of
    ``download.select_component_paths`` for the HF path.

    ``exclude`` drops the named component subfolders (th#1330 B2). An
    exclusion that matches nothing is a no-op, not an error: the caller
    derives it from the dispatch's component overrides, and an override whose
    component is not a subfolder of the base tree is a load-time refusal
    (``ComponentSubstitutionError``), not a fetch-time one."""
    paths = [f.path for f in res.files]
    if components and not components_present(paths, components):
        raise ValueError(
            f"components= {list(components)!r} matched nothing in "
            f"{ref.canonical()} snapshot {res.snapshot_digest[:16]}"
        )
    keep = select_component_paths(paths, components, exclude)
    files = [f for f in res.files if f.path in keep]
    return WorkerResolvedRepo(snapshot_digest=res.snapshot_digest, files=files)


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

class CozySnapshotDownloader:
    """Downloads blobs into a CAS layout, reassembles chunked files, materializes snapshot.

    Layout under <base_dir>:
      blobs/blake3/<aa>/<bb>/<digest>
      snapshots/<snapshot_digest>/...            (whole repo)
      snapshots/<snapshot_digest>__c<fp>/...     (components=-scoped subset, pgw#505)
    """

    def __init__(self) -> None:
        pass

    async def ensure_snapshot(
        self,
        base_dir: Path,
        ref: TensorhubRef,
        *,
        resolved: Optional[WorkerResolvedRepo],
        progress: Optional[ProgressFn] = None,
        components: Sequence[str] = (),
        exclude_components: Sequence[str] = (),
        fill_source_dir: Optional[Path] = None,
    ) -> Path:
        """``fill_source_dir`` (th#850 managed-tier ruling, gw#599): an
        endpoint-scoped datacenter-warm CAS root (RunPod volume mount, same
        ``blobs/`` layout as ``base_dir``) consulted before R2 on a blob
        miss. ``None`` is the degenerate cozy-local/no-volume case — fetch
        goes straight to R2, byte-identical to pre-th#850 behavior. Never
        the CAS root itself; ``base_dir`` (local disk) always is."""
        blobs_root = base_dir / "blobs"
        snaps_root = base_dir / "snapshots"
        blobs_root.mkdir(parents=True, exist_ok=True)
        snaps_root.mkdir(parents=True, exist_ok=True)
        fill_blobs_root = fill_source_dir / "blobs" if fill_source_dir is not None else None
        # pgw#972: the volume's chunk-object tree, a SIBLING of `blobs/` — an
        # incomplete file's verified chunks, so a replacement pod resumes it
        # instead of refetching it whole.
        fill_chunks_root = fill_source_dir / "chunks" if fill_source_dir is not None else None

        if resolved is None:
            # Workers don't resolve via HTTP — the orchestrator pre-resolves
            # every cozy ref a job needs and ships URLs via JobExecutionRequest.
            raise RuntimeError(
                "cozy snapshot requires orchestrator-resolved URLs (resolved=None)"
            )
        res = _validate_resolved(ref, resolved)
        if components or exclude_components:
            res = _filter_resolved_components(
                ref, res, components, exclude_components)

        # pgw#505: a components=-scoped fetch materializes a NARROWER tree
        # than the digest's full content, so it is keyed separately (never
        # under the bare digest — that name is reserved for the complete
        # snapshot). th#1330 B2: an exclude_components= scope is narrower in
        # the same sense and keys the same way.
        key = snapshot_dir_key(
            res.snapshot_digest, components, exclude_components)
        snap_dir = snaps_root / key
        trust_key = (str(snaps_root.resolve()), key)
        if snap_dir.exists() and trust_key in _TRUSTED_SNAPSHOTS:
            _log.info("snapshot_cached key=%s", key[:24])
            return snap_dir

        # Coordinate concurrent builders via threading (works across event loops).
        loop = asyncio.get_running_loop()
        with _SNAP_LOCK:
            if snap_dir.exists() and trust_key in _TRUSTED_SNAPSHOTS:
                return snap_dir
            entry = _SNAP_ENTRIES.get(key)
            if entry is None:
                entry = _SnapshotEntry()
                _SNAP_ENTRIES[key] = entry
                is_builder = True
            else:
                is_builder = False

        if not is_builder:
            _log.info("snapshot_waiting key=%s (another builder active)", key[:24])
            await loop.run_in_executor(None, entry.event.wait)
            if entry.exception is not None:
                raise RuntimeError("concurrent snapshot build failed") from entry.exception
            return snap_dir

        # pgw#971: volume write-throughs that could not be teed run in the
        # background and are JOINED below, so the build never returns — or
        # fails — with a publish still in flight.
        publishes: List["asyncio.Task"] = []
        try:
            if snap_dir.exists():
                # gw#598: a materialized tree this process has not produced or
                # checked (another pod's writes on a shared volume root, or a
                # previous boot's) is verified ONCE before reuse; corruption
                # quarantines tree + bad blobs and falls through to a rebuild.
                ok, bad = await asyncio.to_thread(
                    _verify_materialized_tree, snap_dir, res.files
                )
                if ok:
                    _mark_trusted(_TRUSTED_SNAPSHOTS, trust_key)
                    _log.info("snapshot_cached key=%s (verified first use)", key[:24])
                    return snap_dir
                _log.warning(
                    "snapshot_reuse_corrupt key=%s bad_files=%d; quarantining and rebuilding",
                    key[:24], len(bad),
                )
                await asyncio.to_thread(
                    _quarantine_materialized, snap_dir, blobs_root, bad
                )
            _log.info("snapshot_build_start key=%s files=%d", key[:24], len(res.files))
            await self._ensure_blobs(
                blobs_root,
                res.files,
                progress=progress,
                fill_blobs_root=fill_blobs_root,
                fill_chunks_root=fill_chunks_root,
                publishes=publishes,
            )
            # Materialization copies/concatenates multi-GB trees — strictly
            # off the event loop (gw#407: a loop blocked for the duration of
            # a snapshot build cannot answer the hub; under page-cache
            # pressure that IO takes minutes).
            await asyncio.to_thread(
                self._materialize_snapshot, blobs_root, snaps_root, snap_dir, res
            )
            # Every input was verified this process (fresh downloads by the
            # downloader, reused blobs by _blob_trusted), so the freshly
            # materialized tree is trustworthy without a second hash pass.
            _mark_trusted(_TRUSTED_SNAPSHOTS, trust_key)
            _log.info("snapshot_build_done key=%s", key[:24])
            return snap_dir
        except BaseException as exc:
            entry.exception = exc
            raise
        finally:
            # Join the background write-throughs unconditionally: they are
            # best-effort in OUTCOME, never in LIFETIME. `return_exceptions`
            # because a failed publish must not turn a good build into a
            # failure — nor a failed build into a different error.
            if publishes:
                await asyncio.gather(*publishes, return_exceptions=True)
            # Digest-poisoning fix (#358): a FAILED build must not park a
            # set-event + stale exception under this digest forever. Evict the
            # entry so the next request creates a fresh builder and retries;
            # waiters already holding this entry still see its exception once.
            with _SNAP_LOCK:
                if _SNAP_ENTRIES.get(key) is entry:
                    del _SNAP_ENTRIES[key]
            entry.event.set()

    def _materialize_snapshot(
        self,
        blobs_root: Path,
        snaps_root: Path,
        snap_dir: Path,
        res: WorkerResolvedRepo,
    ) -> None:
        """Blocking build phase (worker thread): reassemble + hardlink into a
        writer-unique ``.building-<writer_id>`` dir, then atomically rename
        into place. Writer-unique (not a fixed ``.building`` name) so two
        writers racing to materialize the same snapshot key on the same CAS
        root (concurrent tasks within one pod, or an operator-configured
        shared root) never interleave writes into the same tree — only the
        atomic rename below, not the build, decides
        the winner."""
        writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        tmp = snaps_root / f"{snap_dir.name}.building-{writer_id}"
        tmp.mkdir(parents=True, exist_ok=True)

        self._reassemble_chunked(blobs_root, tmp, res.files)
        self._materialize_regular(blobs_root, tmp, res.files)

        # Atomic rename; a concurrent writer may have already published.
        if snap_dir.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            try:
                tmp.rename(snap_dir)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
                if not snap_dir.exists():
                    raise
            else:
                fsync_dir(snaps_root)  # persist the rename itself (gw#408)

    # ------------------------------------------------------------------
    # Blob download (deduplicated, parallel)
    # ------------------------------------------------------------------

    async def _ensure_blobs(
        self,
        blobs_root: Path,
        files: List[WorkerResolvedRepoFile],
        *,
        progress: Optional[ProgressFn] = None,
        fill_blobs_root: Optional[Path] = None,
        fill_chunks_root: Optional[Path] = None,
        publishes: Optional[List["asyncio.Task"]] = None,
    ) -> None:
        # pgw#971: background volume write-throughs land here; the caller joins
        # them before returning, so none can outlive the build.
        if publishes is None:
            publishes = []
        # Deduplicate by digest — same blob referenced by multiple paths (e.g.
        # fp16 and normal variants sharing the same part) is downloaded once.
        seen: Set[str] = set()
        unique: List[WorkerResolvedRepoFile] = []
        for f in files:
            # th#1303: dedupe on the ALGORITHM-TAGGED ref. Two different
            # algorithms' hex could otherwise collide in this set and one
            # blob would stand in for the other.
            digest = f.cas_ref()
            if not f.url and not f.transfer_grant and not f.chunks:
                raise ValueError(f"missing transfer for {f.path}")
            if digest not in seen:
                seen.add(digest)
                unique.append(f)

        _log.info("ensure_blobs total_entries=%d unique_blobs=%d", len(files), len(unique))

        cached_digests = {
            f.cas_ref() for f in unique
            if _blob_path(blobs_root, f.cas_ref()).exists()
        }
        missing_bytes = sum(
            int(f.size_bytes or 0) for f in unique
            if f.cas_ref() not in cached_digests
        )
        self._check_disk_headroom(blobs_root, missing_bytes)

        total = sum(int(f.size_bytes or 0) for f in unique) or None
        done = total - missing_bytes if total else 0
        network_bytes = 0  # th#850: bytes actually fetched over the network
        # this call, as opposed to bytes already present under blobs_root —
        # the signal a "volume-attached boot ⇒ ~0 network bytes" runtime
        # assertion needs. Only _dl_locked's real fetch increments it.
        done_lock = threading.Lock()
        # Captured HERE, in the caller's context, not looked up per call.
        # Chunked fetches run on a ThreadPoolExecutor created inside
        # chunk_cas, and a bare ThreadPoolExecutor does NOT propagate
        # contextvars — so a per-call `_NETWORK_BYTES_SINK.get()` returns the
        # default None on every chunk and the whole transfer reports ZERO
        # network bytes. th#850's "volume-attached boot ⇒ ~0 network bytes"
        # assertion reads this counter, so that would make every cold chunked
        # boot look warm. The sink is a shared mutable list; binding the
        # reference once is both correct and cheaper.
        _sink = _NETWORK_BYTES_SINK.get()

        def _on_bytes(n: int, *, network: bool = False) -> None:
            nonlocal done, network_bytes
            with done_lock:
                done += n
                if network:
                    network_bytes += n
                    # th#850 managed-tier ruling (gw#599): update the
                    # NetworkBytesScope sink LIVE (not just once at the end
                    # of this call, see below) so a caller's mid-flight
                    # `progress()` tick — called synchronously right below,
                    # still holding nothing but this lock — can read a
                    # genuinely-running total. tensorhub reads network_bytes
                    # off the DOWNLOADING events' running value, the same
                    # way it reads bytes_done/bytes_total.
                    if _sink is not None:
                        _sink[0] += n
                    # ie#522: a real network byte is honest proof the
                    # activity (self-mint compile's load phase, etc.) is
                    # alive — heartbeat it directly, independent of the
                    # CPU-sampling watchdog thread (I/O-bound fills are
                    # CPU-light by design and would otherwise starve it).
                    _activity.note_progress()
                # pgw#1041: fetched bytes as an activity COUNTER, so the 10s
                # beat carries byte-level fetch progress to the hub (the
                # heartbeat above proves liveness but reports no number).
                # Activity-scoped: finished on phase change/activity end
                # (pgw#962), re-acquired per tick.
                act = _activity.current()
                if act is not None:
                    act.counter("download:bytes", "bytes").add(n)
                d = done if total is None else min(done, total)
            if progress is not None:
                try:
                    progress(d, total)
                except Exception:
                    pass

        if progress is not None:
            try:
                progress(min(done, total) if total else done, total)
            except Exception:
                pass

        # Sort largest first for better overlap, then download in parallel.
        unique.sort(key=lambda f: int(f.size_bytes or 0), reverse=True)

        # pgw#1087: per-COMPONENT fetch spans. `weights_fetch` says how long
        # the whole ref took and nothing about which of transformer / vae /
        # text_encoder owned it, and — because these run four-wide — nothing
        # about how much of the wall is genuinely serialized. Both are
        # prerequisites for a trace-overlaps-download optimization, and neither
        # was derivable from anything the platform stored. Boot only: a
        # steady-state materialization hours later must not land in the ladder.
        components: Dict[str, int] = {}
        if boot_phases.in_boot():
            for f in unique:
                components[_component_of(f.path)] = (
                    components.get(_component_of(f.path), 0) + 1)
        # No `ref` here by design: `_ensure_blobs` is blob-level and the ref
        # axis rides the enclosing `weights_fetch` row, which is this span's
        # parent. Repeating it would be a second spelling that can drift.
        comp_spans = (
            boot_phases.ComponentSpans(components) if components else None
        )

        max_conc = 4
        sem = asyncio.Semaphore(max_conc)
        blobs_root_id = str(blobs_root.resolve())

        def _drop_chunks(digest: str) -> None:
            """pgw#972 cleanup, and the whole of it: the volume holds the
            COMPLETE blob under its digest name, so that file's chunk objects
            are garbage. Every site that ESTABLISHES that fact calls this — the
            copy publisher below, the pod that finds it already published, and
            the tee publisher inside `download_chunked_file` — so the only
            chunk sets that survive belong to files no pod has ever completed
            here, and the first pod that completes one collects it. No sweep,
            no owner registry, no age clock."""
            if fill_chunks_root is not None:
                drop_volume_chunks(volume_chunk_dir(fill_chunks_root, digest))

        async def _blob_trusted(dst: Path, f: WorkerResolvedRepoFile, digest: str) -> bool:
            """Reusable AND digest-trusted (gw#598): size gate as before, plus
            a full content-hash check once per (root, digest) per process —
            reused bytes on a shared volume root are another pod's writes of
            any age and must earn the same trust as a fresh verified download.

            The hash DISPATCHES on the ref's algorithm (th#1303); it is never
            skipped, because a check that does not run is indistinguishable
            from one that passes."""
            if not self._blob_usable(dst, f):
                return False
            vkey = (blobs_root_id, digest)
            if vkey in _VERIFIED_BLOBS:
                return True
            algo, want_hex = parse_cas_ref(digest)
            got = await asyncio.to_thread(hash_file, dst, algo)
            if got.lower() == want_hex:
                _mark_trusted(_VERIFIED_BLOBS, vkey)
                return True
            _log.warning(
                "blob_corrupt path=%s digest=%s %s mismatch on reuse; re-downloading",
                f.path, digest[:24], algo,
            )
            dst.unlink(missing_ok=True)
            return False

        async def _fill_from_volume(f: WorkerResolvedRepoFile, digest: str, dst: Path) -> bool:
            """th#850 managed-tier ruling (gw#599): FILL SOURCE #1. A blob
            present on the endpoint volume is BLAKE3-verified and copied to
            local CAS — never trusted on size alone (digest-verification of
            volume-read blobs is mandatory regardless of the local/shared
            distinction, per Paul's ruling)."""
            if fill_blobs_root is None:
                return False
            src = _blob_path(fill_blobs_root, digest)
            started = time.monotonic()
            if await asyncio.to_thread(
                _copy_verified_blob, src, dst, digest, int(f.size_bytes or 0)
            ):
                _on_bytes(int(f.size_bytes or 0))  # not network: volume hit
                _mark_trusted(_VERIFIED_BLOBS, (blobs_root_id, digest))
                _log.info(
                    "blob_cache_hit source=volume digest=%s bytes=%d transfer_ms=%d",
                    digest[:16], int(f.size_bytes or 0),
                    int((time.monotonic() - started) * 1000),
                )
                # Another pod completed this file; any chunk objects a third
                # left behind for it are now dead.
                await asyncio.to_thread(_drop_chunks, digest)
                return True
            _log.info("blob_cache_miss source=volume digest=%s", digest[:16])
            return False

        async def _fill_to_volume(f: WorkerResolvedRepoFile, digest: str, dst: Path) -> None:
            """Write-through (gw#599): a fresh R2 fetch warms the volume for
            the next same-endpoint pod. Best-effort — a publish failure never
            fails the request; the next pod simply falls through to R2 too.

            pgw#971: this is now the FALLBACK. Chunked blobs — which is where a
            multi-GB keep-set's bytes actually live — fill both stores in one
            pass inside ``download_chunked_file``; only whole-file blobs (a v2
            manifest chunks anything over 64 MiB, so these are the small ones)
            and tee failures still pay for a re-read. It also runs in the
            BACKGROUND: the blob's own task completes on download, and
            ``ensure_snapshot`` joins every publish before it returns, so the
            copies overlap each other and the materialization instead of
            sitting on each blob's critical path."""
            if fill_blobs_root is None:
                return
            try:
                published = await asyncio.to_thread(
                    _copy_verified_blob, dst, _blob_path(fill_blobs_root, digest),
                    digest, int(f.size_bytes or 0),
                )
            except Exception as exc:  # noqa: BLE001 - never fails the request
                _log.warning("blob_fill_publish_failed digest=%s: %s", digest[:16], exc)
                return
            _log.info(
                "blob_fill_publish source=r2 destination=volume mode=copy "
                "digest=%s bytes=%d published=%s",
                digest[:16], int(f.size_bytes or 0), published,
            )
            if published:
                await asyncio.to_thread(_drop_chunks, digest)

        async def _dl(f: WorkerResolvedRepoFile) -> None:
            # pgw#1087: the component span brackets THIS file's contribution
            # and carries the source it actually came from — the same
            # cached/volume/network distinction `weights_fetch` reports, one
            # level finer, so "the vae was warm and the transformer was cold"
            # is a row rather than an inference from the total.
            if comp_spans is None:
                await _dl_one(f, _NO_SOURCE)
                return
            component = _component_of(f.path)
            comp_spans.start(component)
            seen: List[str] = []
            try:
                await _dl_one(f, seen.append)
            finally:
                comp_spans.finish(
                    component,
                    bytes_moved=int(f.size_bytes or 0),
                    source=seen[-1] if seen else "")

        async def _dl_one(
            f: WorkerResolvedRepoFile, note: Callable[[str], None],
        ) -> None:
            digest = f.cas_ref()
            dst = _blob_path(blobs_root, digest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if await _blob_trusted(dst, f, digest):
                _log.info("blob_cached path=%s digest=%s", f.path, digest[:16])
                note(boot_phases.SOURCE_LOCAL)
                return
            async with _inflight_blob_lock(digest):
                if await _blob_trusted(dst, f, digest):
                    _log.info("blob_shared_inflight path=%s digest=%s (sibling ref downloaded it)",
                              f.path, digest[:16])
                    note(boot_phases.SOURCE_INFLIGHT_SHARE)
                    return
                if await _fill_from_volume(f, digest, dst):
                    note(boot_phases.SOURCE_VOLUME)
                    return
                teed = await _dl_locked(f, digest, dst)
                note(boot_phases.SOURCE_R2)
            if fill_blobs_root is not None and not teed:
                publishes.append(
                    asyncio.create_task(_fill_to_volume(f, digest, dst))
                )

        async def _dl_locked(f: WorkerResolvedRepoFile, digest: str, dst: Path) -> bool:
            teed = False
            async with sem:
                if await _blob_trusted(dst, f, digest):
                    return False
                _log.info("blob_download_start path=%s size=%s digest=%s chunks=%d",
                          f.path, f.size_bytes, digest[:24], len(f.chunks))
                if f.chunks:
                    # th#1303 v2: bounded out-of-order fetch, IN-ORDER commit,
                    # whole-file hash fused into the commit stream. Only the
                    # chunks are store-enforced, so this is the one place a
                    # wrong whole-file label is caught — it fails closed.
                    specs = [
                        ChunkSpec(sha256=c.sha256, url=c.url, length=int(c.length))
                        for c in f.chunks
                    ]
                    # pgw#971: fill BOTH stores in this one pass when a volume
                    # is attached — the NFS write hides behind network latency
                    # and the extra read+hash pass disappears.
                    teed = await asyncio.to_thread(
                        download_chunked_file,
                        specs,
                        dst,
                        whole_digest=digest,
                        total_size=int(f.size_bytes or 0),
                        chunk_size_bytes=int(f.chunk_size_bytes or 0)
                        or CAS_CHUNK_SIZE_BYTES,
                        on_bytes=lambda n: _on_bytes(n, network=True),
                        mirror_dst=(
                            _blob_path(fill_blobs_root, digest)
                            if fill_blobs_root is not None else None
                        ),
                        # pgw#972: verified chunks land here as they arrive and
                        # are adopted from here by the next pod, so a pod that
                        # dies 90% in costs its successor 10% of the bytes, not
                        # 100%.
                        mirror_chunk_dir=(
                            volume_chunk_dir(fill_chunks_root, digest)
                            if fill_chunks_root is not None else None
                        ),
                    )
                else:
                    # A WHOLE file. th#1303 S1: BOTH transports now take the
                    # algorithm-tagged digest and verify it themselves, so
                    # there is exactly ONE verification and no call site picks
                    # an algorithm. The old shape passed a bare blake3 hex to
                    # whichever transport understood it and re-hashed the file
                    # afterwards for the other — with an `if` in front of each,
                    # i.e. two vacuous guards guarding one set of bytes.
                    parse_cas_ref(digest)  # refuse an undigestable entry early
                    if f.transfer_grant:
                        grant = S3TransferGrant.from_mapping(f.transfer_grant)
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: download_file_with_grant(
                                grant=grant,
                                dest_path=dst,
                                expected_size_bytes=int(f.size_bytes or 0) or None,
                                expected_digest=digest,
                            ),
                        )
                        _on_bytes(int(f.size_bytes or 0), network=True)
                    else:
                        assert f.url is not None  # validated in _ensure_blobs
                        await _download_one_file(
                            f.url,
                            dst,
                            expected_size=int(f.size_bytes or 0),
                            expected_digest=digest,
                            on_bytes=lambda n: _on_bytes(n, network=True),
                        )
                # Every path above verified size + the content digest under the
                # algorithm the manifest named, before publishing.
                _mark_trusted(_VERIFIED_BLOBS, (blobs_root_id, digest))
                _log.info("blob_download_done path=%s digest=%s", f.path, digest[:24])
            return teed

        try:
            await asyncio.gather(*(_dl(f) for f in unique))
        finally:
            # A fetch that raised leaves components mid-flight; close them
            # NAMED rather than leaving open rows the hub cannot interpret.
            if comp_spans is not None:
                comp_spans.close_all("fetch_aborted")
        # th#850 managed-tier runtime-assertion signal: on a volume-attached
        # boot with blobs already warm, network_bytes should land near zero.
        # _on_bytes above already streamed this total into the caller's
        # NetworkBytesScope live, chunk by chunk (so mid-flight DOWNLOADING
        # ticks see a genuine running total, not just the final tally) — this
        # is a log line only, not a second write to the sink.
        cache_hit_pct = 100.0 * (1 - network_bytes / total) if total else 100.0
        _log.info(
            "ensure_blobs_summary total_bytes=%s network_bytes=%d cache_hit_pct=%.1f",
            total, network_bytes, cache_hit_pct,
        )

    @staticmethod
    def _blob_usable(dst: Path, f: WorkerResolvedRepoFile) -> bool:
        """A cached blob is only reusable at the manifest's size (gw#408): a
        truncated blob from a pre-durability build must be re-downloaded, not
        silently rebuilt into every future snapshot."""
        try:
            if not dst.exists():
                return False
            expected = int(f.size_bytes or 0)
            if expected and dst.stat().st_size != expected:
                _log.warning(
                    "blob_corrupt path=%s digest=%s size=%d expected=%d; re-downloading",
                    f.path, f.digest[:24], dst.stat().st_size, expected,
                )
                dst.unlink(missing_ok=True)
                return False
            return True
        except OSError:
            return False

    @staticmethod
    def _check_disk_headroom(blobs_root: Path, missing_bytes: int) -> None:
        if missing_bytes <= 0:
            return
        try:
            free = shutil.disk_usage(blobs_root).free
        except OSError:
            return
        required = missing_bytes + _DISK_HEADROOM_BYTES
        if required > free:
            raise InsufficientDiskError(
                f"insufficient disk for snapshot download: need {required} bytes "
                f"({missing_bytes} blobs + headroom), {free} free at {blobs_root}",
                available_bytes=free,
                required_bytes=required,
                path=str(blobs_root),
            )

    # ------------------------------------------------------------------
    # Chunked file reassembly
    # ------------------------------------------------------------------

    def _reassemble_chunked(
        self, blobs_root: Path, tmp: Path, files: List[WorkerResolvedRepoFile]
    ) -> None:
        """Read .parts.json manifests and concatenate part blobs into original files."""
        for f in files:
            if not _is_parts_manifest(f.path):
                continue

            _log.info("reassemble_start manifest=%s", f.path)
            manifest_blob = _blob_path(blobs_root, f.cas_ref())
            manifest = json.loads(manifest_blob.read_bytes())

            original_path = str(manifest.get("original_path") or "").strip()
            if not original_path:
                raise ValueError(f"parts manifest {f.path} missing original_path")
            parts = manifest.get("parts") or []
            if not parts:
                raise ValueError(f"parts manifest {f.path} has no parts")

            dst = tmp / _norm_rel_path(original_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()

            total_written = 0
            with open(dst, "wb") as out_f:
                for i, part in enumerate(parts):
                    # Keep the ALGORITHM TAG — `_blob_path` parses it. Stripping
                    # one algorithm's prefix here is how a sha256 part would be
                    # looked up under blobs/blake3/, i.e. at nothing.
                    part_digest = str(part.get("digest") or "").strip().lower()
                    if not part_digest:
                        raise ValueError(f"part[{i}] in {f.path} missing digest")
                    part_blob = _blob_path(blobs_root, part_digest)
                    part_size = part_blob.stat().st_size
                    _log.info("  concat part=%d/%d digest=%s size=%s",
                              i + 1, len(parts), part_digest[:16], part_size)
                    with open(part_blob, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    total_written += part_size
                out_f.flush()
                os.fsync(out_f.fileno())  # durable before the snapshot rename (gw#408)

            _log.info("reassemble_done file=%s total_size=%s", original_path, total_written)

    # ------------------------------------------------------------------
    # Regular (non-chunked) file materialization
    # ------------------------------------------------------------------

    def _materialize_regular(
        self, blobs_root: Path, tmp: Path, files: List[WorkerResolvedRepoFile]
    ) -> None:
        """Hardlink/copy non-chunked blobs into the snapshot tree."""
        part_paths = {f.path for f in files if _is_part_file(f.path)}
        for f in files:
            if _is_parts_manifest(f.path) or f.path in part_paths:
                continue
            dst = tmp / _norm_rel_path(f.path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = _blob_path(blobs_root, f.cas_ref())
            _try_hardlink_or_copy(src, dst)



def _verify_materialized_tree(
    snap_dir: Path, files: List[WorkerResolvedRepoFile],
) -> tuple:
    """Integrity of a reused materialized snapshot (worker thread, blocking).

    Manifest-covered regular files are checked against declared size AND their
    CONTENT DIGEST, hashed under the algorithm the manifest named (th#1303);
    reassembled chunked originals (which the manifest digests only part-wise)
    get the structural safetensors check. Returns ``(ok, bad)`` — algorithm-
    tagged refs in ``bad`` name blobs to quarantine.

    The digest check is MANDATORY. It used to read `f.blake3` and guard on that
    field being non-empty, which under manifest v2 is always empty — so a v2
    tree was reported clean having hashed nothing."""

    bad: List[str] = []
    covered: Set[Path] = set()
    for f in files:
        if _is_parts_manifest(f.path) or _is_part_file(f.path):
            continue  # not materialized: parts live only in blobs/
        try:
            dst = snap_dir / _norm_rel_path(f.path)
        except ValueError:
            continue
        covered.add(dst)
        ref = ""
        try:
            ref = f.cas_ref()
            if not dst.exists():
                raise ValueError("missing")
            if f.size_bytes and dst.stat().st_size != int(f.size_bytes):
                raise ValueError("size mismatch")
            verify_file_digest(dst, ref)
        except (OSError, ValueError) as exc:
            _log.warning("snapshot reuse file %s/%s corrupt: %s", snap_dir.name, f.path, exc)
            bad.append(ref or f.path)
    try:
        candidates = sorted(snap_dir.rglob("*.safetensors"))
    except OSError:
        candidates = []
    for st in candidates:
        if st in covered:
            continue
        if not safetensors_file_valid(st):
            _log.warning("snapshot reuse file %s structurally invalid (truncated?)", st)
            bad.append(str(st.relative_to(snap_dir)))
    return (not bad, bad)


def _quarantine_materialized(snap_dir: Path, blobs_root: Path, bad: Any) -> None:
    """Delete a corrupt reused tree AND the corrupt blobs it links, so the
    rebuild re-downloads clean bytes instead of re-linking the same rot."""
    shutil.rmtree(snap_dir, ignore_errors=True)
    for raw in bad or ():
        # Keep the ALGORITHM TAG: stripping it here would aim the unlink at
        # blobs/blake3/<sha256hex>, i.e. at nothing, and leave the corrupt
        # blob in place to be re-linked by the next materialization.
        try:
            _blob_path(blobs_root, str(raw or "")).unlink(missing_ok=True)
        except (OSError, ValueError):
            continue  # path-shaped entry (structural failure), not a digest
    fsync_dir(snap_dir.parent)


def delete_blobs(base_dir: Path, digests: Any) -> None:
    """Remove specific CAS blobs (quarantine of digest-mismatched content,
    gw#408) so a re-materialization re-downloads them instead of re-linking
    the same corrupt bytes."""
    blobs_root = Path(base_dir) / "blobs"
    for raw in digests or ():
        try:
            _blob_path(blobs_root, str(raw or "")).unlink(missing_ok=True)
        except (OSError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

async def ensure_snapshot_async(
    *,
    base_dir: Path,
    ref: TensorhubRef,
    resolved: Optional[WorkerResolvedRepo],
    progress: Optional[ProgressFn] = None,
    components: Sequence[str] = (),
    exclude_components: Sequence[str] = (),
    fill_source_dir: Optional[Path] = None,
) -> Path:
    dl = CozySnapshotDownloader()
    return await dl.ensure_snapshot(
        base_dir, ref, resolved=resolved, progress=progress, components=components,
        exclude_components=exclude_components,
        fill_source_dir=fill_source_dir,
    )
