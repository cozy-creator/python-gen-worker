"""The real fill path, driven from a test, with a fault-injectable origin.

pgw#1632. Every `ensure_*` on the weight path is supposed to be a pure function
of (manifest, store): present objects are skipped, absent ones are fetched, and
a second call — or a re-entry after a kill — costs nothing but the scan. That
property has never had an instrument, which is why pgw#1596 (a headroom gate
that demanded the whole tree while 82% of it was resident) reached a $2/hr H200
before anything noticed.

Everything here is the production code path. The only injected part is the
BYTES' ORIGIN: a real HTTP server that answers real grant URLs and can be told
to stop answering after N bytes, which is how "the fetcher raised at k%" is
expressed without mocking a single one of the functions under test.

The disk budget is injected the same way production allows it to be —
:class:`ModelStore`'s own ``disk_free_bytes_fn`` constructor parameter — over a
REAL measurement of the REAL files the fill wrote, so "the tmpfs is 1x the tree"
is enforced arithmetic rather than a mounted filesystem no CI runner can create.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Iterable, Optional, Sequence

from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef, WireRef
from gen_worker.models.store import ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _Refused(Exception):
    pass


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        origin: "Origin" = self.server.origin  # type: ignore[attr-defined]
        key = self.path.rsplit("/", 1)[-1]
        body = origin.blobs.get(key)
        if body is None:
            self.send_error(404)
            return
        try:
            origin.charge(len(body))
        except _Refused:
            # 410 GONE is TERMINAL in `transfer.grants._retry` (any 4xx that is
            # not 408/425/429 refuses without retrying), so the fault surfaces
            # as one immediate failure instead of five backoffs. That is the
            # shape a kill has: the fetch stops, it does not grind.
            self.send_error(410)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Origin:
    """A real HTTP origin for CAS objects, with a byte fuse.

    ``wire_bytes`` is what actually left it — the independent check on every
    claim a fill makes about what it fetched. ``cutoff_bytes`` arms the fuse:
    once serving one more object would cross it, every further GET is refused
    terminally. That is the fault injection, and it is at the only layer this
    harness is allowed to fake.
    """

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.served = 0
        self.cutoff_bytes: Optional[int] = None
        self.refusals = 0
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.origin = self  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def charge(self, n: int) -> None:
        with self.lock:
            if self.cutoff_bytes is not None and self.served + n > self.cutoff_bytes:
                self.refusals += 1
                raise _Refused
            self.served += n

    def put(self, data: bytes) -> str:
        digest = sha(data)
        self.blobs[digest] = data
        host, port = self.server.server_address[0], self.server.server_address[1]
        return f"http://{host!s}:{port!s}/{digest}"

    @property
    def wire_bytes(self) -> int:
        with self.lock:
            return int(self.served)

    def arm(self, cutoff_bytes: Optional[int]) -> None:
        with self.lock:
            self.cutoff_bytes = cutoff_bytes

    def reset(self) -> None:
        with self.lock:
            self.served = 0
            self.refusals = 0
            self.cutoff_bytes = None

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@dataclass(frozen=True)
class Tree:
    """A synthetic checkpoint: distinct objects, known sizes, real URLs."""

    files: tuple[tuple[str, bytes, str], ...]

    @property
    def total_bytes(self) -> int:
        return sum(len(body) for _p, body, _u in self.files)

    @property
    def object_bytes(self) -> int:
        return len(self.files[0][1])

    def resolved(self) -> WorkerResolvedRepo:
        fingerprint = hashlib.sha256(
            b"|".join(f"{p}:{sha(b)}".encode() for p, b, _u in self.files)
        ).hexdigest()
        return WorkerResolvedRepo(
            snapshot_digest="sha256:" + fingerprint,
            files=[
                WorkerResolvedRepoFile(p, len(b), url, digest=sha(b))
                for p, b, url in self.files
            ],
        )

    def snapshot(self) -> pb.Snapshot:
        resolved = self.resolved()
        return pb.Snapshot(
            digest=resolved.snapshot_digest,
            files=[
                pb.SnapshotFile(
                    path=f.path, size_bytes=f.size_bytes, digest=f.digest, url=f.url,
                )
                for f in resolved.files
            ],
        )


def build_tree(origin: Origin, *, objects: int = 32, object_bytes: int = 256 * 1024) -> Tree:
    """A tree of distinct, incompressible-enough objects on a real origin.

    Sized so per-object granularity (1/objects) resolves every k the harness
    asks for, and so the whole matrix stays inside a CI test budget. The ratios
    that decide pass/fail — resident vs missing, high-water vs tree — are size
    independent.
    """

    files: list[tuple[str, bytes, str]] = []
    for i in range(objects):
        body = hashlib.sha256(f"object-{i}".encode()).digest() * (object_bytes // 32)
        assert len(body) == object_bytes
        path = f"component-{i % 4}/shard-{i:03d}.safetensors"
        files.append((path, body, origin.put(body)))
    return Tree(tuple(files))


def disk_used(root: Path) -> int:
    """Bytes the fill has actually put on this disk, counted from the files.

    ``st_blocks`` would count sparseness and symlink targets twice; the
    question here is "how many bytes of tree does this pod hold", which the
    apparent sizes of the regular files answer exactly. Symlinks (the
    projection's publish form) are deliberately not followed — a projected tree
    that points into the CAS is not a second copy, and counting it as one would
    make the high-water assertion unfalsifiable.
    """

    total = 0
    for path in root.rglob("*"):
        try:
            status = path.lstat()
        except OSError:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        total += int(status.st_size)
    return total


class HighWater:
    """Samples real on-disk bytes while an operation runs."""

    def __init__(self, root: Path, interval_s: float = 0.01) -> None:
        self.root = root
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> int:
        try:
            now = disk_used(self.root)
        except OSError:
            return self.peak
        self.peak = max(self.peak, now)
        return now

    def __enter__(self) -> "HighWater":
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="high-water", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample()


class Wire:
    """A bound sink collecting the envelopes a real fill emits."""

    def __init__(self) -> None:
        self.messages: list[pb.WorkerMessage] = []

    async def send(self, msg: pb.WorkerMessage) -> None:
        self.messages.append(msg)

    @property
    def model_events(self) -> list[Any]:
        return [
            m.model_event for m in self.messages
            if m.WhichOneof("msg") == "model_event"
        ]


def reset_fill_memos() -> None:
    """Forget every in-process memo, so the next pass is a fresh PROCESS.

    A re-entry that is only re-entering an in-memory cache proves nothing about
    resumability. The trusted-snapshot set and the in-flight builder map are
    module state; a real pod restart has neither.
    """

    from gen_worker.models import cozy_snapshot

    cozy_snapshot._TRUSTED_SNAPSHOTS.clear()
    with cozy_snapshot._SNAP_LOCK:
        cozy_snapshot._SNAP_ENTRIES.clear()


@dataclass
class FillContext:
    cache_dir: Path
    tree: Tree
    origin: Origin
    ref: WireRef = field(default=WireRef("acme/harness-model"))
    disk_free_bytes_fn: Optional[Callable[[], int]] = None
    wire: Wire = field(default_factory=Wire)

    def store(self) -> ModelStore:
        return ModelStore(
            self.wire.send,
            cache_dir=self.cache_dir,
            disk_free_bytes_fn=self.disk_free_bytes_fn,
        )

    def budget_fn(self, budget_bytes: int) -> Callable[[], int]:
        """Free space on a volume of exactly ``budget_bytes``, measured for real.

        This is the 1x-tree tmpfs, expressed as arithmetic over the bytes the
        fill has genuinely written. A gate that charges for the whole tree on a
        re-entry cannot satisfy it; a gate that charges for the remainder can.
        """

        cache_dir = self.cache_dir

        def free() -> int:
            return max(0, budget_bytes - disk_used(cache_dir))

        return free


#: One fill entry point, as the harness drives it.
@dataclass(frozen=True)
class Op:
    name: str
    run: Callable[[FillContext], Awaitable[Path]]


async def _op_store_ensure_local(ctx: FillContext) -> Path:
    return await ctx.store().ensure_local(ctx.ref, ctx.tree.snapshot())


async def _op_store_materialize_local(ctx: FillContext) -> Path:
    return (
        await ctx.store()._materialize_local(ctx.ref, ctx.tree.snapshot())
    ).path


async def _op_download_ensure_local(ctx: FillContext) -> Path:
    from gen_worker.models import download as download_mod
    from gen_worker.models.store import _snapshot_to_resolved

    return await download_mod.ensure_local(
        str(ctx.ref),
        provider="tensorhub",
        snapshot=_snapshot_to_resolved(ctx.tree.snapshot()),
        cache_dir=ctx.cache_dir,
    )


async def _op_ensure_snapshot_async(ctx: FillContext) -> Path:
    from gen_worker.models.cozy_snapshot import ensure_snapshot_async

    return await ensure_snapshot_async(
        base_dir=ctx.cache_dir,
        ref=TensorhubRef(owner="acme", repo="harness-model", release="latest"),
        resolved=ctx.tree.resolved(),
        progress=None,
        fill_source_dir=None,
    )


#: THE COVERED SET. `tests/test_ensure_idempotence_pgw1632.py` asserts every
#: byte-moving `ensure_*` in `models/` is either here or classified as moving
#: no bytes — a new fill cannot be added without answering the question.
FILL_OPS: tuple[Op, ...] = (
    Op("ModelStore.ensure_local", _op_store_ensure_local),
    Op("ModelStore._materialize_local", _op_store_materialize_local),
    Op("download.ensure_local", _op_download_ensure_local),
    Op("cozy_snapshot.ensure_snapshot_async", _op_ensure_snapshot_async),
)


def run_fill(ctx: FillContext, op: Op) -> Path:
    """One pass of ``op`` on a fresh process view of the store."""

    reset_fill_memos()

    async def go() -> Path:
        from gen_worker import activity

        activity.bind_sink(ctx.wire.send, asyncio.get_running_loop())
        path = await op.run(ctx)
        for _ in range(8):
            await asyncio.sleep(0)
        return path

    return asyncio.run(go())


def resident_bytes(cache_dir: Path, tree: Tree) -> int:
    """Bytes of ``tree`` the pod's CAS actually holds — the fill's own predicate."""

    from gen_worker.models.cache_paths import open_worker_cas

    cas = open_worker_cas(cache_dir)
    total = 0
    for _path, body, _url in tree.files:
        if cas.contains(sha(body), size=len(body)):
            total += len(body)
    return total


def free_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def wait_for(predicate: Callable[[], bool], *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


__all__ = [
    "FILL_OPS",
    "FillContext",
    "HighWater",
    "Op",
    "Origin",
    "Tree",
    "Wire",
    "build_tree",
    "disk_used",
    "free_disk_bytes",
    "reset_fill_memos",
    "resident_bytes",
    "run_fill",
    "sha",
    "wait_for",
]
