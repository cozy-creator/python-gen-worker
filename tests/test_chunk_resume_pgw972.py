"""pgw#972: chunk-granular resume that crosses PODS.

Paul's ruling: the endpoint volume never holds unverified bytes — it holds
individually-VERIFIED chunks of an incomplete file. A successor pod inventories
them, RE-HASHES every one it adopts, and fetches only what is missing, in any
order.

Everything here runs through the real `download_chunked_file` /
`ensure_snapshot_async` against a real threaded HTTP server on localhost: real
sockets, real bodies, real threads, real files on two real directory trees
standing in for local CAS and the volume. Every property under test is a
property of concurrency and IO, so a mock would assert nothing.

The evidence is always a COUNT — which chunks the server was asked for, which
objects exist, which bytes they hold. Never a clock.

Run: pytest tests/test_chunk_resume_pgw972.py -q
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Dict, Iterable, List

import pytest

import gen_worker.models.chunk_cas as cc
from gen_worker.models.chunk_cas import (
    ChunkSpec,
    download_chunked_file,
    drop_volume_chunks,
    volume_chunk_dir,
)
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.refs import TensorhubRef

CS = 8192  # chunk size: the ARITHMETIC is under test, not the volume


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _body(total: int, seed: int = 11) -> bytes:
    out = bytearray(total)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(total):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


# ---------------------------------------------------------------------------
# A real HTTP server serving chunk objects by digest, one URL per chunk —
# which is the production shape: the hub presigns every chunk separately and a
# chunked file has NO whole-file object to range over.
# ---------------------------------------------------------------------------

class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        key = self.path.rsplit("/", 1)[-1]
        with srv.lock:
            srv.hits[key] = srv.hits.get(key, 0) + 1
            blob = srv.blobs.get(key)
            dead = key in srv.dead
        if blob is None:
            self.send_error(404)
            return
        if dead:
            self.close_connection = True
            try:
                self.connection.close()
            except OSError:
                pass
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Shrink the RETRY SCHEDULE, not the policy. The tests that kill a source
    walk the full `_CHUNK_MAX_ATTEMPTS` ladder by design (pgw#972's give-up is a
    COUNT), and at the production cap that is ~45-90s of real sleeping per dead
    chunk. Nothing here asserts on a duration — the schedule is shrunk so the
    COUNTS the tests do assert on are reachable in a suite."""
    monkeypatch.setattr(cc, "_CHUNK_BACKOFF_CAP_S", 0.02)


@pytest.fixture()
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.blobs: Dict[str, bytes] = {}
    srv.hits: Dict[str, int] = {}
    srv.dead: set[str] = set()
    srv.lock = threading.Lock()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _publish(srv, payload: bytes) -> List[ChunkSpec]:
    specs: List[ChunkSpec] = []
    base = f"http://127.0.0.1:{srv.server_address[1]}/chunk"
    for off in range(0, len(payload), CS):
        part = payload[off:off + CS]
        d = _sha(part)
        with srv.lock:
            srv.blobs[d] = part
        specs.append(ChunkSpec(sha256=d, url=f"{base}/{d}", length=len(part)))
    return specs


def _fetched(srv, specs: Iterable[ChunkSpec]) -> Dict[str, int]:
    with srv.lock:
        return {s.sha256: srv.hits.get(s.sha256, 0) for s in specs}


def _bytes_fetched(srv, specs: List[ChunkSpec]) -> int:
    """How many BYTES the source was asked for. The measured quantity resume
    exists to reduce, and it is a count of served bodies, not a duration."""
    hits = _fetched(srv, specs)
    return sum(hits[s.sha256] * s.length for s in specs)


def _seed_volume_chunks(
    chunks_root: Path, whole_digest: str, specs: List[ChunkSpec],
    payload: bytes, indices: Iterable[int],
) -> Path:
    """Stand in for a previous pod that verified and published these chunks."""
    d = volume_chunk_dir(chunks_root, whole_digest)
    d.mkdir(parents=True, exist_ok=True)
    for i in indices:
        (d / f"{i:08d}-{specs[i].sha256}").write_bytes(payload[i * CS:(i + 1) * CS])
    return d


# ---------------------------------------------------------------------------
# (a) A successor pod adopts what the volume holds and fetches only the rest
# ---------------------------------------------------------------------------

def test_a_successor_pod_fetches_only_the_chunks_the_volume_lacks(
    tmp_path: Path, server,
) -> None:
    """The whole point of the issue, driven end to end through TWO real
    downloads: pod 1 dies part-way, pod 2 starts with an EMPTY local CAS and
    the volume pod 1 left behind, and pays only for the missing chunks.

    "A pod that dies 90% into a 35 GB component leaves its successor to refetch
    all 35 GB from R2" — this is that sentence, falsified.
    """
    payload = _body(CS * 8)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"
    chunk_dir = volume_chunk_dir(volume / "chunks", whole)

    # --- pod 1: the LAST chunk's source is dead, so the file never assembles.
    with server.lock:
        server.dead.add(specs[7].sha256)
    with pytest.raises(Exception):
        download_chunked_file(
            specs, tmp_path / "pod1" / "blob", whole_digest=whole,
            total_size=len(payload), chunk_size_bytes=CS, window=2,
            mirror_dst=volume / "blobs" / "blob",
            mirror_chunk_dir=chunk_dir,
        )
    # It published every chunk it VERIFIED, and nothing it did not.
    left = sorted(p.name for p in chunk_dir.iterdir())
    assert left == [f"{i:08d}-{specs[i].sha256}" for i in range(7)], left
    # And no whole-file blob — the local file never proved its digest.
    assert not (volume / "blobs" / "blob").exists()
    after_pod1 = _fetched(server, specs)

    # --- pod 2: a REPLACEMENT pod. Empty local CAS, same volume, live source.
    with server.lock:
        server.dead.clear()
    local2 = tmp_path / "pod2" / "blob"
    published = download_chunked_file(
        specs, local2, whole_digest=whole,
        total_size=len(payload), chunk_size_bytes=CS, window=2,
        mirror_dst=volume / "blobs" / "blob",
        mirror_chunk_dir=chunk_dir,
    )

    assert local2.read_bytes() == payload
    assert published is True
    after_pod2 = _fetched(server, specs)
    # The seven adopted chunks were NOT asked for again; only the missing one.
    for i in range(7):
        assert after_pod2[specs[i].sha256] == after_pod1[specs[i].sha256], i
    assert after_pod2[specs[7].sha256] == after_pod1[specs[7].sha256] + 1


def test_the_adopted_set_may_be_any_subset_in_any_order(
    tmp_path: Path, server,
) -> None:
    """"This can happen out of order too; the chunks of a repo-CAS file do not
    need to be downloaded in some logical order." A non-contiguous adoption is
    the normal case, not a degenerate one — the verified prefix the whole-file
    hasher follows is rebuilt from whatever landed."""
    payload = _body(CS * 6, seed=4)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"
    _seed_volume_chunks(volume / "chunks", whole, specs, payload, [0, 2, 5])

    local = tmp_path / "local" / "blob"
    download_chunked_file(
        specs, local, whole_digest=whole,
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_chunk_dir=volume_chunk_dir(volume / "chunks", whole),
    )

    assert local.read_bytes() == payload
    hits = _fetched(server, specs)
    assert [hits[s.sha256] for s in specs] == [0, 1, 0, 1, 1, 0]


def test_without_a_volume_the_successor_refetches_everything(
    tmp_path: Path, server,
) -> None:
    """The CONTROL arm, and the measurement: the same interrupted-then-resumed
    sequence with no volume attached pays for every byte a second time. This is
    what the numbers in the test above are relative to."""
    payload = _body(CS * 8)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)

    with server.lock:
        server.dead.add(specs[7].sha256)
    with pytest.raises(Exception):
        download_chunked_file(
            specs, tmp_path / "pod1" / "blob", whole_digest=whole,
            total_size=len(payload), chunk_size_bytes=CS, window=2,
        )
    with server.lock:
        server.dead.clear()
    before = _bytes_fetched(server, specs)

    local2 = tmp_path / "pod2" / "blob"
    download_chunked_file(
        specs, local2, whole_digest=whole,
        total_size=len(payload), chunk_size_bytes=CS, window=2,
    )
    assert local2.read_bytes() == payload
    # Eight chunks fetched again, where the volume-backed successor fetched one.
    assert _bytes_fetched(server, specs) - before == len(payload)


# ---------------------------------------------------------------------------
# (b) An adopted chunk is RE-HASHED, so a corrupt one is caught and refetched
# ---------------------------------------------------------------------------

def test_a_corrupted_volume_chunk_is_rejected_on_adopt_and_refetched(
    tmp_path: Path, server, caplog,
) -> None:
    """The name on the volume is a CLAIM, not a proof. Adoption re-hashes every
    range it takes, so a corrupt object costs a wasted hash pass and can never
    inject bytes — and it is unlinked, so the next pod does not pay again."""
    caplog.set_level("WARNING", logger="gen_worker.models.chunk_cas")
    payload = _body(CS * 5, seed=7)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"
    chunk_dir = _seed_volume_chunks(
        volume / "chunks", whole, specs, payload, [0, 1, 2, 3, 4]
    )
    # Rot one object in place: right name, right LENGTH, wrong bytes. A
    # size-only check would wave it straight into the file.
    victim = chunk_dir / f"{2:08d}-{specs[2].sha256}"
    victim.write_bytes(_body(CS, seed=999))

    local = tmp_path / "local" / "blob"
    # No `mirror_dst`, so no whole-file blob is published and the cleanup rule
    # does not fire — this test is about the chunk objects themselves.
    download_chunked_file(
        specs, local, whole_digest=whole,
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_chunk_dir=chunk_dir,
    )

    # The file is right, and exactly one chunk was bought.
    assert local.read_bytes() == payload
    hits = _fetched(server, specs)
    assert [hits[s.sha256] for s in specs] == [0, 0, 1, 0, 0]
    assert "volume_chunk_corrupt" in caplog.text
    # The refetch republished it, so the volume now holds honest bytes.
    assert victim.read_bytes() == payload[CS * 2:CS * 3]


def test_a_truncated_volume_chunk_is_rejected_too(tmp_path: Path, server) -> None:
    """A short object is the ENOSPC shape, and it must not be adopted as a
    partial range — the manifest's length is checked by the hash covering it."""
    payload = _body(CS * 3, seed=8)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    chunk_dir = _seed_volume_chunks(
        tmp_path / "volume" / "chunks", whole, specs, payload, [0, 1, 2]
    )
    (chunk_dir / f"{1:08d}-{specs[1].sha256}").write_bytes(payload[CS:CS + 64])

    local = tmp_path / "local" / "blob"
    download_chunked_file(
        specs, local, whole_digest=whole, total_size=len(payload),
        chunk_size_bytes=CS, window=2, mirror_chunk_dir=chunk_dir,
    )
    assert local.read_bytes() == payload
    hits = _fetched(server, specs)
    assert [hits[s.sha256] for s in specs] == [0, 1, 0]


# ---------------------------------------------------------------------------
# (c) Duplicate concurrent writes of one chunk are harmless
# ---------------------------------------------------------------------------

def test_two_pods_writing_the_same_chunks_at_once_are_harmless(
    tmp_path: Path, server,
) -> None:
    """The name is content-addressed, so two pods that reach it wrote the SAME
    bytes and a duplicate write is a no-op. That — plus writer-unique staging
    and an atomic rename — is what dissolves the two-writers objection to
    putting resume state on shared storage.

    Two REAL concurrent downloads of one file into two local CAS roots (as two
    pods of one endpoint have) sharing one volume chunk directory.
    """
    payload = _body(CS * 6, seed=13)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    chunk_dir = volume_chunk_dir(tmp_path / "volume" / "chunks", whole)

    errors: List[BaseException] = []
    start = threading.Barrier(2)

    def _pod(name: str) -> None:
        try:
            start.wait(timeout=30)
            download_chunked_file(
                specs, tmp_path / name / "blob", whole_digest=whole,
                total_size=len(payload), chunk_size_bytes=CS, window=6,
                mirror_chunk_dir=chunk_dir,
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=_pod, args=(n,)) for n in ("podA", "podB")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, errors
    assert (tmp_path / "podA" / "blob").read_bytes() == payload
    assert (tmp_path / "podB" / "blob").read_bytes() == payload
    # One object per chunk, each holding its own bytes — no doubling, no
    # interleaving, and no staging litter from the loser of any race.
    names = sorted(p.name for p in chunk_dir.iterdir())
    assert names == [f"{i:08d}-{specs[i].sha256}" for i in range(6)], names
    for i in range(6):
        obj = chunk_dir / f"{i:08d}-{specs[i].sha256}"
        assert _sha(obj.read_bytes()) == specs[i].sha256


# ---------------------------------------------------------------------------
# (d) Completed-file cleanup, and its concurrency story
# ---------------------------------------------------------------------------

def test_publishing_the_whole_blob_drops_the_chunk_garbage(
    tmp_path: Path, server,
) -> None:
    """Once the volume holds the COMPLETE blob under its digest name, that
    file's chunk objects are garbage and the publisher removes them. Steady
    state on the volume is therefore exactly what it was before pgw#972 — only
    files still in flight carry chunk objects."""
    payload = _body(CS * 4, seed=21)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"
    chunk_dir = volume_chunk_dir(volume / "chunks", whole)

    published = download_chunked_file(
        specs, tmp_path / "local" / "blob", whole_digest=whole,
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_dst=volume / "blobs" / "blob", mirror_chunk_dir=chunk_dir,
    )

    assert published is True
    assert (volume / "blobs" / "blob").read_bytes() == payload
    assert not chunk_dir.exists()


def test_an_adopter_already_reading_an_object_is_not_racing_the_drop(
    tmp_path: Path, server,
) -> None:
    """`unlink` is a NAMESPACE operation: a reader that already opened an
    object keeps its inode to the end. So a drop concurrent with an adoption
    cannot truncate what that adopter is reading — the worst case anywhere in
    this design is a wasted refetch, never wrong bytes."""
    payload = _body(CS * 3, seed=31)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    chunk_dir = _seed_volume_chunks(
        tmp_path / "volume" / "chunks", whole, specs, payload, [0, 1, 2]
    )

    with open(chunk_dir / f"{1:08d}-{specs[1].sha256}", "rb") as held:
        first = held.read(16)
        assert drop_volume_chunks(chunk_dir) == 3
        assert not chunk_dir.exists()
        # The open handle still sees the whole, correct object.
        assert first + held.read() == payload[CS:CS * 2]

    # An adopter that arrives AFTER the drop simply misses and refetches.
    local = tmp_path / "local" / "blob"
    download_chunked_file(
        specs, local, whole_digest=whole, total_size=len(payload),
        chunk_size_bytes=CS, window=2, mirror_chunk_dir=chunk_dir,
    )
    assert local.read_bytes() == payload
    assert [_fetched(server, specs)[s.sha256] for s in specs] == [1, 1, 1]


def test_the_drop_leaves_an_inflight_writers_staged_object_alone(
    tmp_path: Path,
) -> None:
    """Staged chunks are dot-prefixed and writer-unique; they belong to a live
    writer, so the drop skips them and the `rmdir` fails harmlessly. The next
    pod to complete this file collects the directory."""
    d = tmp_path / "volume" / "chunks" / "sha256" / "aa" / "bb" / ("aa" * 32)
    d.mkdir(parents=True)
    (d / ("00000000-" + "c" * 64)).write_bytes(b"done")
    staged = d / ".00000001-x.part-999-deadbeef"
    staged.write_bytes(b"in flight")

    assert drop_volume_chunks(d) == 1
    assert d.exists() and staged.read_bytes() == b"in flight"
    assert drop_volume_chunks(d) == 0  # nothing left to collect but the stage


def test_dropping_a_directory_that_was_never_created_is_a_no_op(
    tmp_path: Path,
) -> None:
    """Cleanup runs on every path that establishes a complete volume blob,
    including the overwhelmingly common one where no pod ever died."""
    assert drop_volume_chunks(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# The third cleanup site, through the real snapshot path
# ---------------------------------------------------------------------------

def _resolved(payload: bytes, specs: List[ChunkSpec]) -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="sha256:" + ("c7" * 32),
        files=[
            WorkerResolvedRepoFile(
                path="model.safetensors",
                size_bytes=len(payload),
                url=None,
                digest="sha256:" + _sha(payload),
                chunk_size_bytes=CS,
                chunks=tuple(
                    WorkerResolvedChunk(sha256=s.sha256, url=s.url, length=s.length)
                    for s in specs
                ),
            ),
        ],
    )


def test_a_volume_blob_hit_collects_a_dead_pods_chunk_garbage(
    tmp_path: Path, server,
) -> None:
    """The third owner of the cleanup rule: a pod that finds the whole blob
    ALREADY on the volume knows the chunk objects are dead, even though it
    published neither. Driven through `ensure_snapshot_async` so the fill-source
    wiring is the production one."""
    payload = _body(CS * 5, seed=41)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"
    chunk_dir = _seed_volume_chunks(
        volume / "chunks", whole, specs, payload, [0, 1, 2]
    )
    # A previous pod completed the file and published the blob, but died before
    # collecting its chunks (or a third pod's chunks outlived it).
    blob = volume / "blobs" / "sha256" / whole[7:9] / whole[9:11] / whole[7:]
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)

    snap = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path / "local",
        ref=TensorhubRef(owner="org", repo="model"),
        resolved=_resolved(payload, specs),
        fill_source_dir=volume,
    ))

    assert (snap / "model.safetensors").read_bytes() == payload
    # Filled from the volume — not one chunk was bought from the source.
    assert all(v == 0 for v in _fetched(server, specs).values())
    assert not chunk_dir.exists()


def test_a_cold_snapshot_fetch_leaves_no_chunk_garbage_on_the_volume(
    tmp_path: Path, server,
) -> None:
    """End to end on the ordinary path: a cold chunked fetch warms the volume
    with the whole blob and leaves the chunk tree empty behind it."""
    payload = _body(CS * 5, seed=43)
    whole = "sha256:" + _sha(payload)
    specs = _publish(server, payload)
    volume = tmp_path / "volume"

    snap = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path / "local",
        ref=TensorhubRef(owner="org", repo="model"),
        resolved=_resolved(payload, specs),
        fill_source_dir=volume,
    ))

    assert (snap / "model.safetensors").read_bytes() == payload
    blobs = sorted(p.name for p in (volume / "blobs").rglob("*") if p.is_file())
    assert blobs == [whole[7:]]
    assert not volume_chunk_dir(volume / "chunks", whole).exists()
