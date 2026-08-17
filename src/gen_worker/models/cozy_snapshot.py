"""Tensorhub snapshot policy composed over tensorfs.

Tensorhub supplies a resolved manifest and opaque GET grants. tensorfs owns the
chunk manifest, local object store, verified transfers, and materialization.
This module retains only worker policy: component selection, pickle refusal,
disk headroom, endpoint-volume fill order, and boot observability.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import fcntl
import hashlib
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from gen_worker._vendor.tensorfs import (
    CASRef,
    Chunk,
    DigestMismatch,
    FileEntry,
    LocalCAS,
    RefConflict,
    RepositoryManifest,
    TransferGrant,
    download,
)

from .. import activity as _activity
from .. import boot_phases
from ..capability import InsufficientDiskError
from .cache_paths import open_worker_cas
from .download import components_present, select_component_paths
from .errors import (
    PICKLE_WEIGHT_EXTENSIONS,
    PickleWeightRefused,
    first_pickle_weight_path,
)
from .hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from .refs import TensorhubRef

_log = logging.getLogger("gen_worker.download")
ProgressFn = Callable[[int, Optional[int]], None]

_NETWORK_BYTES_SINK: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "cozy_snapshot_network_bytes_sink", default=None
)
_DISK_HEADROOM_BYTES = 1 << 30
_TRUST_CAP = 65_536
_TRUSTED_SNAPSHOTS: set[tuple[str, str]] = set()
_SNAP_LOCK = threading.Lock()
_SNAP_ENTRIES: dict[tuple[str, str], "_SnapshotEntry"] = {}


class NetworkBytesScope:
    def __init__(self) -> None:
        self._sink = [0]
        self._token: contextvars.Token[list[int] | None] | None = None

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


class _SnapshotEntry:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.exception: BaseException | None = None


def _norm_rel_path(path: str) -> str:
    """Tensorhub path adapter; tensorfs performs the authoritative check."""

    value = str(path or "").strip()
    # Constructing an entry applies tensorfs's complete portable-path policy.
    return FileEntry(value, 0, CASRef("0" * 64)).path


def _component_of(path: str) -> str:
    head, separator, _ = _norm_rel_path(path).partition("/")
    return head if separator else "(root)"


# PICKLE_WEIGHT_EXTENSIONS and first_pickle_weight_path moved to .errors
# (pgw#1273) so the HF lane refuses on the same list this one does.


def _manifest_entry(file: WorkerResolvedRepoFile) -> FileEntry:
    digest = CASRef.parse(file.cas_ref())
    chunks = tuple(
        Chunk(CASRef.parse(f"sha256:{chunk.sha256}"), int(chunk.length))
        for chunk in file.chunks
    )
    return FileEntry(
        _norm_rel_path(file.path),
        int(file.size_bytes),
        digest,
        chunks,
    )


def _validate_resolved(ref: TensorhubRef, resolved: WorkerResolvedRepo) -> WorkerResolvedRepo:
    snapshot_digest = str(resolved.snapshot_digest or "").strip()
    if not snapshot_digest:
        raise ValueError("resolved model missing snapshot_digest")

    files: list[WorkerResolvedRepoFile] = []
    for file in resolved.files:
        if not str(file.path or "").strip():
            continue
        entry = _manifest_entry(file)
        if entry.path.endswith(".parts.json") or re.search(r"\.part\d{4}$", entry.path):
            raise ValueError(
                f"resolved model file {entry.path}: legacy split-file snapshots are not "
                "part of tensorfs v1; republish the repository"
            )
        if not file.chunks and not str(file.url or "").strip():
            raise ValueError(f"resolved model file missing transfer: {entry.path}")
        if file.chunks and any(not str(chunk.url or "").strip() for chunk in file.chunks):
            raise ValueError(f"resolved model file missing chunk transfer: {entry.path}")
        files.append(
            WorkerResolvedRepoFile(
                path=entry.path,
                size_bytes=entry.size_bytes,
                url=str(file.url or "").strip() or None,
                digest=str(entry.digest),
                chunks=tuple(file.chunks),
            )
        )

    if not files:
        raise ValueError("resolved model has empty files list")
    if bad := first_pickle_weight_path(file.path for file in files):
        raise PickleWeightRefused(
            f"refusing snapshot {snapshot_digest[:16]} of {ref.canonical()}: "
            f"{bad!r} is a pickle-format weight. Republish with safetensors."
        )
    return WorkerResolvedRepo(snapshot_digest=snapshot_digest, files=files)


SUBSET_MARKER = "__c"


def snapshot_dir_key(
    snapshot_digest: str,
    components: Sequence[str] = (),
) -> str:
    """The snapshot directory name. th#1941: for a hub-composed manifest the
    digest IS the whole key — the composition is already in it. ``components``
    is the author-declared positive subset (HF/Hub ``components=``)."""
    included = sorted({str(value).strip() for value in components if str(value).strip()})
    key = snapshot_digest
    if included:
        key += SUBSET_MARKER + hashlib.sha1("+".join(included).encode()).hexdigest()[:12]
    return key


def _filter_resolved_components(
    ref: TensorhubRef,
    resolved: WorkerResolvedRepo,
    components: Sequence[str],
) -> WorkerResolvedRepo:
    paths = [file.path for file in resolved.files]
    if not components_present(paths, components):
        raise ValueError(
            f"components={list(components)!r} matched nothing in {ref.canonical()} "
            f"snapshot {resolved.snapshot_digest[:16]}"
        )
    keep = select_component_paths(paths, components)
    return WorkerResolvedRepo(
        snapshot_digest=resolved.snapshot_digest,
        files=[file for file in resolved.files if file.path in keep],
    )


def _manifest(files: Sequence[WorkerResolvedRepoFile]) -> RepositoryManifest:
    return RepositoryManifest(tuple(_manifest_entry(file) for file in files))


def _grants(files: Sequence[WorkerResolvedRepoFile]) -> tuple[TransferGrant, ...]:
    grants: dict[CASRef, TransferGrant] = {}
    for file in files:
        entry = _manifest_entry(file)
        candidates: Iterable[TransferGrant]
        if entry.chunks:
            pairs = zip(entry.chunks, file.chunks, strict=True)
            candidates = (
                TransferGrant(chunk.digest, chunk.length, remote.url)
                for chunk, remote in pairs
            )
        else:
            candidates = (
                TransferGrant(entry.digest, entry.size_bytes, str(file.url or "")),
            )
        for grant in candidates:
            prior = grants.get(grant.digest)
            if prior is not None and prior.size_bytes != grant.size_bytes:
                raise ValueError(f"conflicting grants for {grant.digest}")
            grants.setdefault(grant.digest, grant)
    return tuple(grants.values())


def _file_matches(path: Path, entry: FileEntry) -> bool:
    try:
        if path.stat().st_size != entry.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        return bool(digest.hexdigest() == entry.digest.digest)
    except OSError:
        return False


def _tree_matches(path: Path, manifest: RepositoryManifest) -> bool:
    try:
        actual = {
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_file()
        }
    except OSError:
        return False
    expected = {entry.path for entry in manifest.files}
    return actual == expected and all(
        _file_matches(path / entry.path, entry) for entry in manifest.files
    )


def _materialize_repository(
    cas: LocalCAS, manifest: RepositoryManifest, target: Path
) -> Path:
    """Publish or converge on another process's exact winning tree."""
    try:
        return cas.materialize_repository(manifest, target)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        if _tree_matches(target, manifest):
            return target
        # The winner is not this manifest. Never delete a tree another process
        # may own; preserve the publication collision as the failure.
        raise


def _publish_snapshot(
    cas: LocalCAS, manifest: RepositoryManifest, target: Path
) -> Path:
    """Revalidate and publish one target under its process-shared lock."""

    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if target.exists():
                if _tree_matches(target, manifest):
                    return target
                # This exact target is protected by the flock. Never remove a
                # tree based on validation performed before acquiring it.
                shutil.rmtree(target)
            return _materialize_repository(cas, manifest, target)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _pin_manifest(cas: LocalCAS, key: str, manifest: RepositoryManifest) -> None:
    target = cas.store_manifest(manifest)
    name = f"snapshot:{key}"
    current = cas.read_ref(name)
    if current == target:
        return
    try:
        cas.compare_and_swap_ref(name, target, expected=current)
    except RefConflict:
        if cas.read_ref(name) != target:
            raise


class CozySnapshotDownloader:
    async def ensure_snapshot(
        self,
        base_dir: Path,
        ref: TensorhubRef,
        *,
        resolved: WorkerResolvedRepo | None,
        progress: ProgressFn | None = None,
        components: Sequence[str] = (),
        fill_source_dir: Path | None = None,
    ) -> Path:
        if resolved is None:
            raise RuntimeError("cozy snapshot requires orchestrator-resolved URLs")
        selected = _validate_resolved(ref, resolved)
        if components:
            selected = _filter_resolved_components(ref, selected, components)
        manifest = _manifest(selected.files)
        key = snapshot_dir_key(selected.snapshot_digest, components)
        snapshots = Path(base_dir) / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        target = snapshots / key
        trust_key = (str(snapshots.resolve()), key)
        if target.exists() and trust_key in _TRUSTED_SNAPSHOTS:
            return target

        loop = asyncio.get_running_loop()
        with _SNAP_LOCK:
            entry = _SNAP_ENTRIES.get(trust_key)
            if entry is None:
                entry = _SnapshotEntry()
                _SNAP_ENTRIES[trust_key] = entry
                builder = True
            else:
                builder = False
        if not builder:
            await loop.run_in_executor(None, entry.event.wait)
            if entry.exception is not None:
                raise RuntimeError("concurrent snapshot build failed") from entry.exception
            return target

        try:
            if target.exists():
                if await asyncio.to_thread(_tree_matches, target, manifest):
                    await asyncio.to_thread(
                        _pin_manifest, open_worker_cas(base_dir), key, manifest
                    )
                    if len(_TRUSTED_SNAPSHOTS) >= _TRUST_CAP:
                        _TRUSTED_SNAPSHOTS.clear()
                    _TRUSTED_SNAPSHOTS.add(trust_key)
                    return target

            await self._ensure_objects(
                open_worker_cas(base_dir),
                selected.files,
                progress=progress,
                fill=(
                    open_worker_cas(fill_source_dir)
                    if fill_source_dir is not None
                    else None
                ),
            )
            cas = open_worker_cas(base_dir)
            await asyncio.to_thread(_pin_manifest, cas, key, manifest)
            await asyncio.to_thread(_publish_snapshot, cas, manifest, target)
            if len(_TRUSTED_SNAPSHOTS) >= _TRUST_CAP:
                _TRUSTED_SNAPSHOTS.clear()
            _TRUSTED_SNAPSHOTS.add(trust_key)
            return target
        except BaseException as exc:
            entry.exception = exc
            raise
        finally:
            with _SNAP_LOCK:
                if _SNAP_ENTRIES.get(trust_key) is entry:
                    del _SNAP_ENTRIES[trust_key]
            entry.event.set()

    async def _ensure_objects(
        self,
        cas: LocalCAS,
        files: Sequence[WorkerResolvedRepoFile],
        *,
        progress: ProgressFn | None,
        fill: LocalCAS | None,
    ) -> None:
        grants = _grants(files)
        entries = {file.path: _manifest_entry(file) for file in files}
        pending = {
            path: {ref for ref, _size in entry.objects()}
            for path, entry in entries.items()
        }
        sources = {path: boot_phases.SOURCE_LOCAL for path in entries}
        components: dict[str, int] = {}
        if boot_phases.in_boot():
            for path in entries:
                component = _component_of(path)
                components[component] = components.get(component, 0) + 1
        comp_spans = boot_phases.ComponentSpans(components) if components else None
        if comp_spans is not None:
            for path in entries:
                comp_spans.start(_component_of(path))
        settle_lock = threading.Lock()

        def settle(ref: CASRef, source: str) -> None:
            finished: list[str] = []
            with settle_lock:
                for path, outstanding in pending.items():
                    if ref not in outstanding:
                        continue
                    outstanding.remove(ref)
                    if source == boot_phases.SOURCE_R2:
                        sources[path] = source
                    elif (
                        source == boot_phases.SOURCE_VOLUME
                        and sources[path] == boot_phases.SOURCE_LOCAL
                    ):
                        sources[path] = source
                    if not outstanding:
                        finished.append(path)
            if comp_spans is not None:
                for path in finished:
                    comp_spans.finish(
                        _component_of(path),
                        bytes_moved=entries[path].size_bytes,
                        source=sources[path],
                    )

        missing: list[TransferGrant] = []
        done = 0
        total = sum(grant.size_bytes for grant in grants)

        # Report per grant: a counter frozen at zero reads as a stalled
        # transfer and the pod is killed mid-progress.
        def report_residency() -> None:
            if progress is not None:
                progress(done, total)

        def resident(grant: TransferGrant) -> bool:
            try:
                return bool(cas.contains(grant.digest, size=grant.size_bytes))
            except DigestMismatch:
                return False

        def filled(grant: TransferGrant) -> bool:
            if fill is None:
                return False
            try:
                source = fill.verify_object(grant.digest, size=grant.size_bytes)
                cas.put_file(source, expected=grant.digest, size=grant.size_bytes)
                return True
            except (DigestMismatch, FileNotFoundError, OSError):
                return False

        # Every check re-hashes the object — ~1.0s of solid CPU per GiB WARM,
        # but that is the fast end of a ~14x spread: measured cold-and-contended
        # it is ~14s/GiB (tensorfs#42), which is what puts a large resident
        # snapshot past the hub's window. A resume re-hashes everything earlier
        # attempts landed. On the caller's
        # loop thread that stranded the heartbeat and every queued event until
        # the scan ended, so each check runs off-thread and only the emission
        # stays on the caller's thread.
        report_residency()
        for grant in grants:
            if await asyncio.to_thread(resident, grant):
                done += grant.size_bytes
                settle(grant.digest, boot_phases.SOURCE_LOCAL)
            elif await asyncio.to_thread(filled, grant):
                done += grant.size_bytes
                settle(grant.digest, boot_phases.SOURCE_VOLUME)
            else:
                missing.append(grant)
            report_residency()

        # Every resident snapshot occupies disk TWICE: the CAS objects plus the
        # materialized tree `_publish_snapshot` writes next. Sizing only the
        # missing objects under-counted by exactly one whole model, and the
        # `missing and` guard skipped the check entirely when every object was
        # already resident — the case where the publish is the ONLY writer. A
        # pod passed this gate and then ENOSPC'd mid-materialize.
        fetch = sum(grant.size_bytes for grant in missing)
        publish = sum(entry.size_bytes for entry in entries.values())
        required = fetch + publish + _DISK_HEADROOM_BYTES
        free = shutil.disk_usage(cas.root).free
        if required > free:
            raise InsufficientDiskError(
                f"insufficient disk for snapshot: need {required} bytes "
                f"({fetch} to fetch, {publish} to publish), "
                f"{free} free at {cas.root}",
                available_bytes=free,
                required_bytes=required,
                path=str(cas.root),
            )

        sink = _NETWORK_BYTES_SINK.get()
        moved = 0
        progress_lock = threading.Lock()

        def on_object(_digest: CASRef, size: int) -> None:
            nonlocal moved
            with progress_lock:
                moved += size
                if sink is not None:
                    sink[0] += size
                current = moved
            _activity.note_progress()
            if progress is not None:
                progress(min(done + current, total), total)
            settle(_digest, boot_phases.SOURCE_R2)

        try:
            if missing:
                report = await asyncio.to_thread(
                    download, tuple(missing), cas, progress=on_object
                )
                if not report.ok:
                    reasons = [detail for _digest, detail in report.failures]
                    if report.expired:
                        reasons.append(
                            f"expired grants: {', '.join(map(str, report.expired))}"
                        )
                    raise RuntimeError(
                        "tensorfs download failed: " + "; ".join(reasons)
                    )
        except BaseException:
            if comp_spans is not None:
                comp_spans.close_all("fetch_aborted")
            raise

        if fill is not None:
            for grant in missing:
                try:
                    source = cas.verify_object(grant.digest, size=grant.size_bytes)
                    fill.put_file(source, expected=grant.digest, size=grant.size_bytes)
                except (DigestMismatch, OSError):
                    _log.warning("tensorfs volume fill failed for %s", grant.digest)


def delete_blobs(base_dir: Path, digests: Any) -> None:
    cas = open_worker_cas(base_dir)
    for raw in digests or ():
        try:
            cas.object_path(CASRef.parse(str(raw))).unlink(missing_ok=True)
        except (OSError, ValueError):
            continue


async def ensure_snapshot_async(
    *,
    base_dir: Path,
    ref: TensorhubRef,
    resolved: WorkerResolvedRepo | None,
    progress: ProgressFn | None = None,
    components: Sequence[str] = (),
    fill_source_dir: Path | None = None,
) -> Path:
    return await CozySnapshotDownloader().ensure_snapshot(
        base_dir,
        ref,
        resolved=resolved,
        progress=progress,
        components=components,
        fill_source_dir=fill_source_dir,
    )


__all__ = [
    "NetworkBytesScope",
    "PICKLE_WEIGHT_EXTENSIONS",
    "delete_blobs",
    "ensure_snapshot_async",
    "first_pickle_weight_path",
    "snapshot_dir_key",
]
