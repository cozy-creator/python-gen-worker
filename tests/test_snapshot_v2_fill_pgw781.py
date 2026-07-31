"""pgw#781 / th#1303: the worker can FILL a manifest-v2 (sha256 / chunked)
snapshot, and every integrity check on that path actually runs.

Before this, ``cozy_snapshot`` was blake3-only end to end: ``_blob_path``
hardcoded ``blobs/blake3/…``, ``chunk_cas`` was not imported by the fill path
at all, and a v2 entry died at ``missing blake3 for <path>``. So a v2 publish
was unconsumable — which makes "publish v2" untestable as an outcome.

These drive the REAL ``ensure_snapshot_async`` over a real localhost HTTP
server: sockets, threads, files, the actual reassembly. The assertions are
about BYTES and about HOW MANY BYTES WERE HASHED, never about "ok" — a
verifier that examines nothing returns a result byte-identical to one that
passes, which is the exact defect class this whole program keeps finding.

Run: pytest tests/test_snapshot_v2_fill_pgw781.py -q
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import threading
from pathlib import Path

import pytest
from blake3 import blake3

import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.models.chunk_cas import DigestMismatch
from gen_worker.models.cozy_snapshot import (
    NetworkBytesScope,
    _verify_materialized_tree,
    ensure_snapshot_async,
)
from gen_worker.models.hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.refs import TensorhubRef

# Small enough that the tests move kilobytes; the production constant is
# asserted elsewhere. What matters here is the ARITHMETIC and the dispatch.
CS = 4096
_SNAPSHOT = "sha256:" + "a1" * 32


def body(total: int, seed: int = 7) -> bytes:
    out = bytearray(total)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(total):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# A real HTTP server. Chunks are served by digest, so a URL is genuinely
# content-addressed and a permuted chunk list still fetches real chunks.
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        key = self.path.rsplit("/", 1)[-1]
        with srv.lock:
            srv.hits[key] = srv.hits.get(key, 0) + 1
            blob = srv.blobs.get(key)
        if blob is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


@pytest.fixture()
def store():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.blobs, srv.hits, srv.lock = {}, {}, threading.Lock()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def put(store, blob: bytes) -> str:
    """Store bytes under their sha256 and return the URL."""
    d = sha(blob)
    with store.lock:
        store.blobs[d] = blob
    return f"{store.base}/{d}"


def chunked_entry(store, path: str, data: bytes, chunk_size: int = CS):
    """Publish `data` as chunks and return the v2 resolved entry."""
    chunks = []
    for off in range(0, len(data), chunk_size):
        piece = data[off : off + chunk_size]
        chunks.append(
            WorkerResolvedChunk(sha256=sha(piece), url=put(store, piece), length=len(piece))
        )
    return WorkerResolvedRepoFile(
        path=path,
        size_bytes=len(data),
        blake3="",  # v2 leaves the legacy mirror EMPTY — that is the trap
        url=None,  # and a chunked file has NO whole-file URL
        digest="sha256:" + sha(data),
        chunks=tuple(chunks),
        chunk_size_bytes=chunk_size,
    )


def whole_entry(store, path: str, data: bytes):
    return WorkerResolvedRepoFile(
        path=path,
        size_bytes=len(data),
        blake3="",
        url=put(store, data),
        digest="sha256:" + sha(data),
    )


def fill(tmp_path: Path, files, sub: str = "local") -> Path:
    return asyncio.run(
        ensure_snapshot_async(
            base_dir=tmp_path / sub,
            ref=TensorhubRef(owner="org", repo="model"),
            resolved=WorkerResolvedRepo(snapshot_digest=_SNAPSHOT, files=list(files)),
        )
    )


# ---------------------------------------------------------------------------
# The outcome: a v2 snapshot materializes, byte-identically
# ---------------------------------------------------------------------------


def test_v2_chunked_snapshot_materializes_byte_identically(tmp_path, store):
    big = body(CS * 3 + 17)  # 4 chunks, last one short
    small = body(200, seed=2)  # under the threshold: stored whole
    snap = fill(
        tmp_path,
        [
            chunked_entry(store, "transformer/weights.safetensors", big),
            whole_entry(store, "config.json", small),
        ],
    )

    assert (snap / "transformer" / "weights.safetensors").read_bytes() == big
    assert (snap / "config.json").read_bytes() == small
    # Every chunk fetched exactly once — reassembly is not re-downloading.
    assert sorted(store.hits.values()) == [1, 1, 1, 1, 1]


def test_chunks_land_under_the_sha256_root_not_the_blake3_one(tmp_path, store):
    """The algorithm is a PATH SEGMENT. A sha256 blob under blobs/blake3/ would
    collide with a blake3 blob of different bytes at the same hex."""
    data = body(CS * 2)
    fill(tmp_path, [chunked_entry(store, "w.safetensors", data)])
    blobs = tmp_path / "local" / "blobs"
    assert (blobs / "sha256").is_dir()
    assert not (blobs / "blake3").exists()
    d = sha(data)
    assert (blobs / "sha256" / d[:2] / d[2:4] / d).read_bytes() == data


def test_v1_blake3_snapshot_still_fills_unchanged(tmp_path, store):
    """DUAL READ. Every pre-migration checkpoint must keep resolving, and its
    on-disk layout must not move (a moved layout is a cold re-download of the
    entire corpus on every pod)."""
    data = body(900, seed=5)
    b3 = blake3(data).hexdigest()
    with store.lock:
        store.blobs[b3] = data
    entry = WorkerResolvedRepoFile(
        path="model.safetensors",
        size_bytes=len(data),
        blake3=b3,  # legacy: bare hex, no tag, no digest field
        url=f"{store.base}/{b3}",
    )
    snap = fill(tmp_path, [entry], sub="v1")
    assert (snap / "model.safetensors").read_bytes() == data
    blobs = tmp_path / "v1" / "blobs"
    assert (blobs / "blake3" / b3[:2] / b3[2:4] / b3).read_bytes() == data
    assert not (blobs / "sha256").exists()


def test_v1_and_v2_entries_in_one_snapshot(tmp_path, store):
    """The migration window: a repo mid-repoint carries both spellings."""
    v2 = body(CS * 2 + 5)
    v1 = body(300, seed=9)
    b3 = blake3(v1).hexdigest()
    with store.lock:
        store.blobs[b3] = v1
    snap = fill(
        tmp_path,
        [
            chunked_entry(store, "new.safetensors", v2),
            WorkerResolvedRepoFile(
                path="old.safetensors", size_bytes=len(v1), blake3=b3,
                url=f"{store.base}/{b3}",
            ),
        ],
    )
    assert (snap / "new.safetensors").read_bytes() == v2
    assert (snap / "old.safetensors").read_bytes() == v1


# ---------------------------------------------------------------------------
# The checks must actually RUN — revert-turns-red
# ---------------------------------------------------------------------------


def test_lying_whole_file_digest_on_a_chunked_file_refuses_to_install(tmp_path, store):
    """Only the CHUNKS are store-enforced. The whole-file label is the one
    thing the store cannot vouch for, so the fused reassembly hash is the only
    place a wrong label is caught — and it must fail closed."""
    data = body(CS * 2 + 3)
    entry = chunked_entry(store, "w.safetensors", data)
    lying = WorkerResolvedRepoFile(
        path=entry.path, size_bytes=entry.size_bytes, blake3="", url=None,
        digest="sha256:" + sha(b"something else entirely"),
        chunks=entry.chunks, chunk_size_bytes=CS,
    )
    with pytest.raises(DigestMismatch):
        fill(tmp_path, [lying])
    # And nothing was published under the lie.
    d = sha(b"something else entirely")
    assert not (tmp_path / "local" / "blobs" / "sha256" / d[:2] / d[2:4] / d).exists()


def test_wrong_whole_file_digest_on_a_SMALL_v2_file_refuses(tmp_path, store):
    """The ≤64 MiB path has no chunk enforcement at all — it is served by the
    legacy transports, which only know blake3. Before this change a sha256
    whole file went through them with an EMPTY blake3 expectation and was
    accepted unverified."""
    data = body(500, seed=3)
    url = put(store, data)
    lying = WorkerResolvedRepoFile(
        path="config.json", size_bytes=len(data), blake3="", url=url,
        digest="sha256:" + sha(b"not these bytes"),
    )
    with pytest.raises(DigestMismatch):
        fill(tmp_path, [lying])


def test_a_permuted_chunk_list_is_caught(tmp_path, store):
    """Every chunk individually verifies — only the fused whole-file hash can
    catch an order swap. This is the test that a per-chunk check is not
    mistaken for a whole-file check."""
    data = body(CS * 3)
    entry = chunked_entry(store, "w.safetensors", data)
    swapped = (entry.chunks[1], entry.chunks[0]) + tuple(entry.chunks[2:])
    permuted = WorkerResolvedRepoFile(
        path=entry.path, size_bytes=entry.size_bytes, blake3="", url=None,
        digest=entry.digest, chunks=swapped, chunk_size_bytes=CS,
    )
    with pytest.raises(DigestMismatch):
        fill(tmp_path, [permuted])


def test_reuse_verification_hashes_a_v2_tree(tmp_path, store):
    """``_verify_materialized_tree`` guarded on ``f.blake3`` being non-empty.
    Under v2 it always is empty, so a v2 tree was reported CLEAN having hashed
    zero bytes. Assert the opposite direction: a same-size one-byte corruption
    must be caught, which is only possible if the bytes were read."""
    data = body(CS * 2)
    entry = chunked_entry(store, "w.safetensors", data)
    snap = fill(tmp_path, [entry])

    ok, bad = _verify_materialized_tree(snap, [entry])
    assert ok and not bad

    target = snap / "w.safetensors"
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # same size, one byte different
    target.unlink()  # break the hardlink; corrupt the tree copy only
    target.write_bytes(bytes(raw))

    ok, bad = _verify_materialized_tree(snap, [entry])
    assert not ok
    assert bad == [entry.digest]  # ALGORITHM-TAGGED, so quarantine can find it


def test_quarantine_unlinks_the_sha256_blob(tmp_path, store):
    """A stripped tag would aim the unlink at blobs/blake3/<sha256hex> — at
    nothing — leaving the corrupt blob to be re-linked by the next fill."""
    data = body(CS * 2, seed=11)
    entry = chunked_entry(store, "w.safetensors", data)
    snap = fill(tmp_path, [entry])
    blobs = tmp_path / "local" / "blobs"
    d = sha(data)
    blob = blobs / "sha256" / d[:2] / d[2:4] / d
    assert blob.exists()

    snap_mod._quarantine_materialized(snap, blobs, [entry.digest])
    assert not blob.exists()


def test_an_entry_with_no_readable_digest_is_refused(tmp_path, store):
    """Fail closed. An unreadable manifest must never degrade into an
    unverified download — that is how "no digest" becomes "no check"."""
    data = body(100)
    entry = WorkerResolvedRepoFile(
        path="x.json", size_bytes=len(data), blake3="", url=put(store, data),
    )
    with pytest.raises(ValueError, match="no digest"):
        fill(tmp_path, [entry])


def test_untagged_v2_digest_is_a_hard_error(tmp_path, store):
    """A bare hex in the `digest` field must not be guessed at. Guessing is how
    a sha256 gets hashed with blake3."""
    data = body(100, seed=4)
    entry = WorkerResolvedRepoFile(
        path="x.json", size_bytes=len(data), blake3="", url=put(store, data),
        digest=sha(data),  # untagged
    )
    with pytest.raises(ValueError, match="untagged"):
        fill(tmp_path, [entry])


def test_chunk_lengths_must_sum_to_the_declared_size(tmp_path, store):
    """Refused BEFORE any byte moves — a shape lie costs zero transfer."""
    data = body(CS * 2)
    entry = chunked_entry(store, "w.safetensors", data)
    lying = WorkerResolvedRepoFile(
        path=entry.path, size_bytes=len(data) + 1000, blake3="", url=None,
        digest=entry.digest, chunks=entry.chunks, chunk_size_bytes=CS,
    )
    with pytest.raises(ValueError, match="chunk lengths sum"):
        fill(tmp_path, [lying])
    assert store.hits == {}  # nothing was fetched


def test_network_bytes_counts_chunked_transfer(tmp_path, store):
    """The volume-attached-boot assertion (th#850) reads this counter; a
    chunked fill that reported zero would make a cold boot look warm."""
    data = body(CS * 3 + 11)
    with NetworkBytesScope() as scope:
        fill(tmp_path, [chunked_entry(store, "w.safetensors", data)])
    assert scope.network_bytes == len(data)


# ---------------------------------------------------------------------------
# The REST wire shape, as the hub actually serializes it. These exist because
# the first version of this parser was written from a prose contract instead of
# from the hub's serializer, shipped in 0.78.0, and made every real chunked
# checkpoint unparseable — a v2 publish that looked fine on the hub and was
# unreachable to every standalone client. Unit tests all passed: they encoded
# the same wrong assumption.
# ---------------------------------------------------------------------------


def _entry(n_chunks=2, **over):
    e = {
        "path": "model.safetensors", "size_bytes": 100,
        "digest": "sha256:" + "a" * 64,
        "chunk_size_bytes": 64,
        "chunks": [{"digest": f"{i:02d}" + "b" * 62, "len": 50} for i in range(n_chunks)],
        "chunk_urls": [f"https://r2.invalid/{i}" for i in range(n_chunks)],
    }
    e.update(over)
    return e


def _parse(entry):
    from gen_worker.models.hub_client import parse_chunk_list
    return parse_chunk_list("tensorhub resolve for o/r", entry["path"],
                            entry.get("chunks"), entry.get("chunk_urls"))


def test_chunk_urls_is_a_SEPARATE_index_aligned_array():
    """`catalog.SnapshotManifestFile` cannot put URLs inside `chunks` — those
    bytes ARE the content-addressed identity. The URLs ride alongside."""
    got = _parse(_entry())
    assert [c.url for c in got] == ["https://r2.invalid/0", "https://r2.invalid/1"]
    assert [c.sha256[:2] for c in got] == ["00", "01"]
    assert [c.length for c in got] == [50, 50]


def test_a_nested_url_still_works_for_the_grpc_shape():
    """The proto's ChunkRef carries sha256/url/len together, so both forms
    must parse — one client library, two transports."""
    e = _entry()
    e["chunk_urls"] = None
    for i, c in enumerate(e["chunks"]):
        c["url"] = f"https://grpc.invalid/{i}"
    got = _parse(e)
    assert [c.url for c in got] == ["https://grpc.invalid/0", "https://grpc.invalid/1"]


def test_a_MISALIGNED_url_list_is_fatal():
    """Index alignment is the only thing binding a URL to its digest. A short
    list must never silently mean 'fetch fewer chunks'."""
    from gen_worker.models.hub_client import HubResolveError
    e = _entry(n_chunks=3)
    e["chunk_urls"] = ["https://r2.invalid/0"]
    with pytest.raises(HubResolveError, match="index alignment"):
        _parse(e)


def test_a_chunk_with_no_url_anywhere_is_fatal():
    """Unreachable bytes must fail loudly at parse, not at byte 0 of a fetch."""
    from gen_worker.models.hub_client import HubResolveError
    e = _entry()
    e["chunk_urls"] = None
    with pytest.raises(HubResolveError, match="missing digest/url/len"):
        _parse(e)


# ---------------------------------------------------------------------------
# The generated stubs must actually carry v2. This is the guard for the defect
# that made the PRODUCTION path silently non-functional: the hub's proto grew
# `digest`/`chunk_size_bytes`/`chunks` at th#1303 checkpoint 3 and only the GO
# side was regenerated, so the worker's vendored copy still described the v1
# message. Every v2 snapshot then arrived over gRPC with no digest and no
# chunks — fail-closed, but entirely dark, and no test noticed because the
# tests built the dataclass directly and never crossed the wire.
# ---------------------------------------------------------------------------


def test_the_generated_snapshotfile_stub_carries_v2():
    from gen_worker.pb import worker_scheduler_pb2 as pb

    names = [f.name for f in pb.SnapshotFile().DESCRIPTOR.fields]
    for want in ("digest", "chunk_size_bytes", "chunks"):
        assert want in names, (
            f"SnapshotFile has no {want!r} — proto/worker_scheduler.proto is stale "
            f"against the hub. Run `task proto`. Fields: {names}"
        )


def test_the_chunk_message_carries_digest_url_and_len():
    from gen_worker.pb import worker_scheduler_pb2 as pb

    chunks = next(f for f in pb.SnapshotFile().DESCRIPTOR.fields if f.name == "chunks")
    names = [f.name for f in chunks.message_type.fields]
    assert names == ["sha256", "url", "len"], names


def test_the_grpc_conversion_carries_chunks_through():
    """Drive the REAL wire-boundary conversion on a REAL protobuf message.

    `_snapshot_to_resolved` is the only place production snapshots are typed,
    and it is exactly where a stale stub is invisible."""
    from gen_worker.executor import _snapshot_to_resolved
    from gen_worker.pb import worker_scheduler_pb2 as pb

    snap = pb.Snapshot(digest="sha256:" + "c" * 64)
    f = snap.files.add()
    f.path, f.size_bytes, f.digest, f.chunk_size_bytes = (
        "w.safetensors", 130, "sha256:" + "d" * 64, 64)
    for i, (h, n) in enumerate(((("a" * 64), 64), (("b" * 64), 64), (("c" * 64), 2))):
        c = f.chunks.add()
        c.sha256, c.url, c.len = h, f"https://r2.invalid/{i}", n

    out = _snapshot_to_resolved(snap)
    got = out.files[0]
    assert got.digest == "sha256:" + "d" * 64
    assert got.cas_ref() == "sha256:" + "d" * 64
    assert [c.length for c in got.chunks] == [64, 64, 2]
    assert [c.url for c in got.chunks] == [f"https://r2.invalid/{i}" for i in range(3)]
    assert got.chunk_size_bytes == 64
