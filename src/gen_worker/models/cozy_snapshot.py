"""Tensorhub snapshot policy composed over tensorfs."""

from __future__ import annotations

import asyncio
import contextvars
import fcntl
import hashlib
import logging
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from gen_worker._vendor.tensorfs import (
    MAX_CHUNK_SIZE,
    CASRef,
    Chunk,
    DigestMismatch,
    FileEntry,
    LocalCAS,
    RefConflict,
    RepositoryManifest,
    project_snapshot,
)
from gen_worker.transfer.grants import DEFAULT_PARALLEL, TransferGrant, download

from .. import activity as _activity
from .. import boot_phases
from .. import snapshot_pull
from ..capability import InsufficientDiskError
from ..snapshot_pull import SnapshotPullStats
from . import fill_plan, projection
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

    value = str(path or "").strip()
    return FileEntry(value, 0, CASRef("0" * 64)).path


def _component_of(path: str) -> str:
    head, separator, _ = _norm_rel_path(path).partition("/")
    return head if separator else "(root)"


def _manifest_entry(file: WorkerResolvedRepoFile) -> FileEntry:
    digest = CASRef.parse(file.cas_ref())
    chunks = tuple(
        Chunk(CASRef.parse(f"sha256:{chunk.sha256}"), int(chunk.length))
        for chunk in file.chunks
    )
    path = _norm_rel_path(file.path)
    size = int(file.size_bytes)
    # The 64 MiB chunkless ceiling is THIS repo's, enforced here: upstream tensorfs accepts a chunkless entry of any size, but in this fleet an object above 64 MiB cannot exist — tensorhub's publish-v2 lane promotes with a single PUT and refuses anything larger. A hub snapshot describing a large file as ONE object names bytes nothing can hold; accepting it would pin and advertise a tree whose bytes are unreachable.
    if not chunks and size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"resolved model file {path}: {size} bytes with no chunks. An "
            f"object above {MAX_CHUNK_SIZE} bytes cannot be promoted by "
            f"tensorhub's single-PUT publish lane, so this manifest names an "
            f"object that can never be resident (pgw#1366)"
        )
    return FileEntry(path, size, digest, chunks)


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
    """The snapshot directory name."""
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



def _scan_fanout(
    account: Callable[[TransferGrant], None],
    grants: Sequence[TransferGrant],
) -> None:
    width = max(1, min(DEFAULT_PARALLEL, len(grants)))
    if width == 1:
        for grant in grants:
            account(grant)
        return
    with ThreadPoolExecutor(
        max_workers=width, thread_name_prefix="fill-scan"
    ) as pool:
        for future in [pool.submit(account, grant) for grant in grants]:
            future.result()

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


def _entry_matches(path: Path, entry: FileEntry) -> bool:

    if projection.is_projection_artifact(path):
        return (
            projection.projection_fault(
                path, digest=entry.digest.digest, size=entry.size_bytes
            )
            is None
        )
    return _file_matches(path, entry)


def _tree_matches(path: Path, manifest: RepositoryManifest) -> bool:
    try:
        actual = {
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_file() or candidate.is_symlink()
        }
    except OSError:
        return False
    expected = {entry.path for entry in manifest.files}
    return actual == expected and all(
        _entry_matches(path / entry.path, entry) for entry in manifest.files
    )


def _publish_snapshot(
    cas: LocalCAS,
    manifest: RepositoryManifest,
    target: Path,
    *,
    symlinks: bool,
) -> Path:

    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if target.exists():
                if _tree_matches(target, manifest):
                    return target
                shutil.rmtree(target)
            return project_snapshot(cas, manifest, target, symlinks=symlinks)
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

            symlinks = await asyncio.to_thread(
                projection.symlinks_supported, snapshots
            )
            started = time.monotonic()
            stats = await self._ensure_objects(
                open_worker_cas(base_dir),
                selected.files,
                progress=progress,
                fill=(
                    open_worker_cas(fill_source_dir)
                    if fill_source_dir is not None
                    else None
                ),
                publish_bytes=projection.projection_write_bytes(
                    manifest, symlinks=symlinks
                ),
            )
            snapshot_pull.emit_pull(
                stats,
                snapshot=selected.snapshot_digest,
                key=key,
                components=len(components),
                duration_ms=int(round((time.monotonic() - started) * 1000)),
            )
            cas = open_worker_cas(base_dir)
            await asyncio.to_thread(_pin_manifest, cas, key, manifest)
            await asyncio.to_thread(
                _publish_snapshot, cas, manifest, target, symlinks=symlinks
            )
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
        publish_bytes: int,
    ) -> SnapshotPullStats:
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

        def report_residency() -> None:
            if progress is not None:
                progress(done, total)

        def resident(grant: TransferGrant) -> bool:
            # pgw#1631: THE ONE SKIP PREDICATE, shared with the headroom gate's
            # plan (`fill_plan.plan_for_snapshot`). A second spelling here is
            # how pgw#1596 happened: the gate and the fill each derived "what
            # is missing" and drifted.
            return fill_plan.is_present(
                cas, str(grant.digest), int(grant.size_bytes)
            )

        def filled(grant: TransferGrant) -> bool:
            if fill is None:
                return False
            try:
                source = fill.object_path(CASRef.parse(grant.digest))
                cas.put_file(source, expected=grant.digest, size=grant.size_bytes)
                return True
            except (DigestMismatch, FileNotFoundError, OSError, ValueError):
                return False

        resident_objects = 0
        resident_bytes = 0
        filled_objects = 0
        filled_bytes = 0
        report_residency()

        tally_lock = threading.Lock()

        def account(grant: TransferGrant) -> None:
            nonlocal done, resident_objects, resident_bytes
            nonlocal filled_objects, filled_bytes
            # The verdict is taken OUTSIDE the lock — it is the expensive part, and holding a mutex across a 64 MiB hash would rebuild the serial loop this replaces with extra steps.
            if resident(grant):
                source = boot_phases.SOURCE_LOCAL
            elif filled(grant):
                source = boot_phases.SOURCE_VOLUME
            else:
                source = ""
            with tally_lock:
                if source == boot_phases.SOURCE_LOCAL:
                    done += grant.size_bytes
                    resident_objects += 1
                    resident_bytes += grant.size_bytes
                elif source == boot_phases.SOURCE_VOLUME:
                    done += grant.size_bytes
                    filled_objects += 1
                    filled_bytes += grant.size_bytes
                else:
                    missing.append(grant)
                # Inside the lock so a slower thread can never publish a position an earlier one already passed: the hub advances on STRICT INCREASE, and a regressing position reads as a wedge.
                report_residency()
            if source:
                settle(grant.digest, source)

        if grants:
            await asyncio.to_thread(_scan_fanout, account, grants)

        fetch = sum(grant.size_bytes for grant in missing)
        publish = publish_bytes
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

        fetched_objects = 0
        fetched_bytes = 0
        late_resident_objects = 0
        try:
            if missing:
                report = await asyncio.to_thread(
                    download, tuple(missing), cas, progress=on_object
                )
                fetched_objects = report.succeeded
                fetched_bytes = report.bytes_transferred
                late_resident_objects = report.skipped_resident
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

        return SnapshotPullStats(
            requested_objects=len(grants),
            tree_bytes=total,
            fetched_objects=fetched_objects,
            fetched_bytes=fetched_bytes,
            resident_objects=resident_objects,
            resident_bytes=resident_bytes,
            filled_objects=filled_objects,
            filled_bytes=filled_bytes,
            late_resident_objects=late_resident_objects,
        )


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
