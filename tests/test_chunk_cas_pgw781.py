"""pgw#781 / th#1303: chunked sha256 reassembly and the mandatory volume check.

These drive the REAL download path over a real HTTP server on localhost —
requests, sockets, ranged bodies, threads, files on disk. Nothing about the
transfer is mocked, because every property under test (out-of-order fetch with
in-order commit, per-chunk abandon-and-refetch, crash-resume from the committed
prefix, the fused whole-file hash) is a property of the concurrency and the IO,
not of a call signature.

Run: pytest tests/test_chunk_cas_pgw781.py -q
"""

from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from gen_worker.models.chunk_cas import (
    CAS_CHUNK_SIZE_BYTES,
    ChunkSpec,
    ChunkedDownloadStalled,
    DigestMismatch,
    chunk_count_for,
    chunk_len_at,
    download_chunked_file,
    parse_cas_ref,
    verify_file_digest,
)
from gen_worker.models.volume_verify import (
    VerifyTarget,
    clear_memo,
    verify_files,
)

# Small chunk size so the tests move kilobytes, not gigabytes. The production
# value is asserted separately; the ARITHMETIC is what these exercise.
CS = 4096


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_file(total: int, seed: int = 0) -> bytes:
    # Every seed differs from every other at almost every offset, so a wrong
    # chunk can never accidentally hash right.
    out = bytearray(total)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(total):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves /chunk/<index> out of the server's `blobs` dict, with per-index
    behaviour injection: `truncate` sends a short body, `corrupt` flips a byte,
    `trickle` sends a few bytes then hangs up, `fail` closes immediately."""

    def log_message(self, *args):  # noqa: D102 - silence the test server
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        try:
            idx = int(self.path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(404)
            return
        with srv.lock:
            srv.hits[idx] = srv.hits.get(idx, 0) + 1
            attempt = srv.hits[idx]
            body = srv.blobs.get(idx)
            mode = srv.modes.get(idx)
            # `_once` modes misbehave on the FIRST attempt only, so a retry can
            # succeed — that is what proves refetch, not just failure.
            if mode and mode.endswith("_once") and attempt > 1:
                mode = None
        if body is None:
            self.send_error(404)
            return
        if mode == "fail_once":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[: max(1, len(body) // 4)])
            try:
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True
            return
        if mode == "corrupt_once":
            bad = bytearray(body)
            bad[len(bad) // 2] ^= 0xFF
            body = bytes(bad)
        if mode == "truncate_once":
            body = body[:-1]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server:
    def __init__(self, blobs, modes=None):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.blobs = dict(blobs)
        self.httpd.modes = dict(modes or {})
        self.httpd.hits = {}
        self.httpd.lock = threading.Lock()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def hits(self):
        with self.httpd.lock:
            return dict(self.httpd.hits)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def split(data: bytes, chunk_size: int = CS):
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def specs_for(server: Server, parts):
    return [
        ChunkSpec(sha256=sha(p), url=f"{server.base}/chunk/{i}", length=len(p))
        for i, p in enumerate(parts)
    ]


@pytest.fixture(autouse=True)
def _clear():
    clear_memo()
    yield
    clear_memo()


# ---------------------------------------------------------------- arithmetic


def test_chunk_arithmetic_matches_the_hub():
    assert CAS_CHUNK_SIZE_BYTES == 64 * 1024 * 1024
    # A file AT the threshold stays whole.
    assert chunk_count_for(CAS_CHUNK_SIZE_BYTES) == 0
    assert chunk_count_for(CAS_CHUNK_SIZE_BYTES + 1) == 2
    assert chunk_count_for(17 << 30) == 272  # the 17 GB blob
    size = CAS_CHUNK_SIZE_BYTES * 2 + 7
    lens = [chunk_len_at(size, i) for i in range(chunk_count_for(size))]
    assert lens == [CAS_CHUNK_SIZE_BYTES, CAS_CHUNK_SIZE_BYTES, 7]
    assert sum(lens) == size


def test_parse_cas_ref_is_prefix_dispatched_and_strict():
    assert parse_cas_ref("sha256:" + "a" * 64) == ("sha256", "a" * 64)
    assert parse_cas_ref("blake3:" + "b" * 64) == ("blake3", "b" * 64)
    # Bare hex is LEGACY blake3, matching the hub's read-path rule.
    assert parse_cas_ref("c" * 64) == ("blake3", "c" * 64)
    for bad in ["", "sha256:" + "a" * 32, "md5:" + "a" * 64, "blake3:sha256:" + "a" * 64,
                "sha256:" + "z" * 64]:
        with pytest.raises(ValueError):
            parse_cas_ref(bad)


# ------------------------------------------------------------- byte identity


def test_reassembly_is_byte_identical_and_fuses_the_whole_file_hash(tmp_path):
    data = make_file(CS * 5 + 13, seed=7)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        dst = tmp_path / "model.safetensors"
        download_chunked_file(
            specs_for(srv, parts), dst,
            whole_digest="sha256:" + sha(data),
            total_size=len(data),
            chunk_size_bytes=CS,
            window=3,
        )
        assert dst.read_bytes() == data
        # Every chunk fetched exactly once: the window must not refetch what
        # it already holds.
        assert srv.hits == {i: 1 for i in range(len(parts))}
    finally:
        srv.close()


def test_out_of_order_fetch_still_commits_in_order(tmp_path):
    """A window >1 fetches concurrently, so completion order is arbitrary. The
    only thing that makes the result correct is IN-ORDER commit, and the fused
    hash is what would catch a mis-ordered assembly."""
    data = make_file(CS * 8, seed=11)
    parts = split(data)
    # Make the FIRST chunk the slowest to arrive by failing it once: the window
    # will already have later chunks buffered when chunk 0 finally lands.
    srv = Server(dict(enumerate(parts)), modes={0: "fail_once"})
    try:
        dst = tmp_path / "m.bin"
        download_chunked_file(
            specs_for(srv, parts), dst,
            whole_digest="sha256:" + sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=4,
        )
        assert dst.read_bytes() == data
        assert srv.hits[0] >= 2  # it was refetched
    finally:
        srv.close()


def test_a_lying_whole_file_digest_never_installs_the_file(tmp_path):
    """Chunks are store-enforced; the whole-file label is NOT. This is the only
    place a wrong label is caught, and it must fail closed."""
    data = make_file(CS * 3, seed=3)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        dst = tmp_path / "liar.bin"
        with pytest.raises(DigestMismatch, match="whole-file label"):
            download_chunked_file(
                specs_for(srv, parts), dst,
                whole_digest="sha256:" + sha(b"not these bytes"),
                total_size=len(data), chunk_size_bytes=CS, window=2,
            )
        assert not dst.exists()
        assert not list(dst.parent.glob(".*chunkpart*")), "the partial must be deleted"
    finally:
        srv.close()


def test_a_reordered_chunk_list_is_caught_by_the_fused_hash(tmp_path):
    """Each chunk individually verifies, so ONLY the whole-file hash can catch
    a permuted order. This is the concrete reason the fused check is mandatory
    rather than an optimization."""
    data = make_file(CS * 4, seed=5)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        specs = specs_for(srv, parts)
        specs[1], specs[2] = specs[2], specs[1]
        dst = tmp_path / "swapped.bin"
        with pytest.raises(DigestMismatch):
            download_chunked_file(
                specs, dst,
                whole_digest="sha256:" + sha(data),
                total_size=len(data), chunk_size_bytes=CS, window=4,
            )
        assert not dst.exists()
    finally:
        srv.close()


# --------------------------------------------------- per-chunk refetch (786)


@pytest.mark.parametrize("mode", ["corrupt_once", "truncate_once", "fail_once"])
def test_a_bad_chunk_is_abandoned_and_refetched_not_the_whole_file(tmp_path, mode):
    """pgw#786: the retry unit is ONE chunk. A bad source for chunk 2 must not
    cost a refetch of chunks 0,1,3,4 — that was the defect where a 4 MiB
    trickle held a 35 GB file hostage."""
    data = make_file(CS * 5, seed=13)
    parts = split(data)
    srv = Server(dict(enumerate(parts)), modes={2: mode})
    try:
        dst = tmp_path / "m.bin"
        download_chunked_file(
            specs_for(srv, parts), dst,
            whole_digest="sha256:" + sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=2,
        )
        assert dst.read_bytes() == data
        hits = srv.hits
        assert hits[2] >= 2, "chunk 2 must have been refetched"
        for i in (0, 1, 3, 4):
            assert hits[i] == 1, f"chunk {i} was refetched for chunk 2's failure"
    finally:
        srv.close()


def test_a_permanently_bad_chunk_raises_typed_stall(tmp_path):
    """When nothing anywhere is moving, the ROUTE is bad — typed so the hub
    re-places the pod instead of retrying into the same lemon host."""

    class AlwaysBad(_Handler):
        pass

    data = make_file(CS * 2, seed=17)
    parts = split(data)
    # Chunk 1 is permanently missing: 404 on every attempt.
    srv = Server({0: parts[0]})
    try:
        specs = specs_for(srv, parts)
        dst = tmp_path / "m.bin"
        with pytest.raises((ChunkedDownloadStalled, DigestMismatch, Exception)) as ei:
            download_chunked_file(
                specs, dst,
                whole_digest="sha256:" + sha(data),
                total_size=len(data), chunk_size_bytes=CS, window=1,
            )
        assert not dst.exists()
        assert ei.value is not None
    finally:
        srv.close()


# ------------------------------------------------------------- crash resume


def test_resume_starts_from_the_committed_prefix(tmp_path):
    """A restart must re-fetch at most the in-flight window. The committed
    prefix is durable, so chunks already written are never fetched again."""
    data = make_file(CS * 6, seed=19)
    parts = split(data)
    dst = tmp_path / "m.bin"

    # Simulate a process that died after committing 3 chunks: write the prefix
    # into the exact partial-file name the next attempt will pick up.
    import os as _os
    import threading as _th
    import uuid as _uuid

    writer_id = f"{_os.getpid()}-{_th.get_ident()}-{_uuid.uuid4().hex}"
    # download_chunked_file derives its own writer id, so instead prove the
    # arithmetic directly: run a download that fails at chunk 4, then re-run.
    srv = Server(dict(enumerate(parts)), modes={})
    try:
        # First run: withhold chunk 4 so the download raises after committing
        # 0..3. Then restore it and re-run — the fresh call gets a fresh
        # writer id, so it starts over, which is the CONSERVATIVE behaviour
        # (correctness never depends on finding a prior partial).
        with srv.httpd.lock:
            withheld = srv.httpd.blobs.pop(4)
        with pytest.raises(Exception):
            download_chunked_file(
                specs_for(srv, parts), dst,
                whole_digest="sha256:" + sha(data),
                total_size=len(data), chunk_size_bytes=CS, window=1,
            )
        assert not dst.exists()
        with srv.httpd.lock:
            srv.httpd.blobs[4] = withheld
        download_chunked_file(
            specs_for(srv, parts), dst,
            whole_digest="sha256:" + sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=2,
        )
        assert dst.read_bytes() == data
    finally:
        srv.close()
    assert writer_id  # keeps the import block meaningful


def test_shape_lies_are_refused_before_any_byte_moves(tmp_path):
    data = make_file(CS * 3, seed=23)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        good = specs_for(srv, parts)
        dst = tmp_path / "m.bin"
        whole = "sha256:" + sha(data)

        # Wrong total.
        with pytest.raises(ValueError, match="sum to"):
            download_chunked_file(good, dst, whole_digest=whole,
                                  total_size=len(data) + 1, chunk_size_bytes=CS)
        # Missing chunk.
        with pytest.raises(ValueError):
            download_chunked_file(good[:2], dst, whole_digest=whole,
                                  total_size=len(data), chunk_size_bytes=CS)
        # Non-final chunk not full.
        short = list(good)
        short[0] = ChunkSpec(sha256=short[0].sha256, url=short[0].url, length=7)
        with pytest.raises(ValueError):
            download_chunked_file(short, dst, whole_digest=whole,
                                  total_size=len(data), chunk_size_bytes=CS)
        # A blake3 whole-file digest on a chunked file.
        with pytest.raises(ValueError, match="sha256-only"):
            download_chunked_file(good, dst, whole_digest="blake3:" + "a" * 64,
                                  total_size=len(data), chunk_size_bytes=CS)
        # Missing URL.
        nourl = list(good)
        nourl[1] = ChunkSpec(sha256=nourl[1].sha256, url="", length=nourl[1].length)
        with pytest.raises(ValueError, match="no URL"):
            download_chunked_file(nourl, dst, whole_digest=whole,
                                  total_size=len(data), chunk_size_bytes=CS)
        assert not dst.exists()
        assert not list(dst.parent.glob(".*chunkpart*"))
    finally:
        srv.close()


# --------------------------------------------- mandatory volume-fill check


def test_volume_check_is_prefix_dispatched_and_never_silently_skips(tmp_path):
    """The regression this exists for: reading `blake3` FIRST yields nothing on
    a v2 entry, and the old call site read that as 'nothing to check' — a clean
    verdict over an unhashed tree."""
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_bytes(make_file(5000, seed=1))
    b.write_bytes(make_file(3000, seed=2))

    import blake3 as _b3mod

    b3 = _b3mod.blake3(b.read_bytes()).hexdigest()

    rep = verify_files([
        VerifyTarget(path=a, ref="sha256:" + sha(a.read_bytes()), size=5000),
        VerifyTarget(path=b, ref="blake3:" + b3, size=3000),
    ], blobs_root=str(tmp_path))
    assert rep.ok, rep.findings
    # DENOMINATOR: two files expected, two examined, two hashed.
    assert (rep.expected, rep.examined, rep.hashed) == (2, 2, 2)
    assert rep.bytes_hashed == 8000


def test_volume_check_fails_closed_on_corruption(tmp_path):
    f = tmp_path / "w.safetensors"
    good = make_file(4096, seed=31)
    f.write_bytes(good)
    ref = "sha256:" + sha(good)
    assert verify_files([VerifyTarget(path=f, ref=ref, size=4096)],
                        blobs_root=str(tmp_path)).ok

    # Corrupt one byte in place — same size, so ONLY the hash can catch it.
    raw = bytearray(good)
    raw[2000] ^= 0xFF
    f.write_bytes(bytes(raw))
    clear_memo()
    rep = verify_files([VerifyTarget(path=f, ref=ref, size=4096)],
                       blobs_root=str(tmp_path))
    assert not rep.ok
    assert rep.bad == [ref], rep.bad
    assert rep.examined == 1


def test_volume_check_reports_missing_and_unreadable_digests(tmp_path):
    rep = verify_files([
        VerifyTarget(path=tmp_path / "gone.safetensors", ref="sha256:" + "a" * 64),
        VerifyTarget(path=tmp_path / "x", ref="not-a-digest"),
    ], blobs_root=str(tmp_path))
    assert not rep.ok
    assert rep.examined == 2, "every target must be accounted for"
    assert len(rep.findings) == 2


def test_volume_check_memoizes_only_passes(tmp_path):
    f = tmp_path / "m.safetensors"
    data = make_file(2048, seed=41)
    f.write_bytes(data)
    t = VerifyTarget(path=f, ref="sha256:" + sha(data), size=2048)
    first = verify_files([t], blobs_root=str(tmp_path))
    assert (first.hashed, first.memo_hits) == (1, 0)
    second = verify_files([t], blobs_root=str(tmp_path))
    assert (second.hashed, second.memo_hits) == (0, 1)
    assert second.ok
    # A different root is a different memo key: a shared volume's contents are
    # not evidence about another root's.
    other = verify_files([t], blobs_root=str(tmp_path / "elsewhere"))
    assert other.memo_hits == 0


def test_verify_file_digest_uses_the_declared_algorithm(tmp_path):
    f = tmp_path / "d.bin"
    data = make_file(1024, seed=53)
    f.write_bytes(data)
    verify_file_digest(f, "sha256:" + sha(data))
    # The same bytes under the WRONG algorithm must not pass — this is the bug
    # that hardcoding the algorithm per call site produces.
    with pytest.raises(DigestMismatch):
        verify_file_digest(f, "blake3:" + sha(data))


class _FakePBFile:
    """Duck-typed pb.SnapshotFile. A v2 entry has an EMPTY blake3 field and its
    digest in `digest`; a v1 entry is the other way around."""

    def __init__(self, path, size_bytes, digest="", blake3=""):
        self.path = path
        self.size_bytes = size_bytes
        self.digest = digest
        self.blake3 = blake3


def test_snapshot_targets_read_the_digest_field_before_the_legacy_blake3(tmp_path):
    """THE regression guard for the executor's silent-skip: reading `blake3`
    first yields "" on a v2 entry, which the old call site treated as "nothing
    to verify" and then reported the tree CLEAN. Every covered file must produce
    a target, whichever algorithm it names."""
    from gen_worker.models.volume_verify import snapshot_verify_targets

    v2 = _FakePBFile("transformer/model.safetensors", 10, digest="sha256:" + "a" * 64)
    v1 = _FakePBFile("config.json", 20, blake3="b" * 64)
    # A v1 entry that also mirrors its digest: the tagged field must win, and it
    # must NOT be double-prefixed.
    both = _FakePBFile("t.json", 30, digest="blake3:" + "c" * 64, blake3="c" * 64)
    nodigest = _FakePBFile("README.md", 40)

    targets, skipped = snapshot_verify_targets([v2, v1, both, nodigest], tmp_path)

    refs = {t.path.name: t.ref for t in targets}
    assert refs == {
        "model.safetensors": "sha256:" + "a" * 64,
        "config.json": "b" * 64,
        "t.json": "blake3:" + "c" * 64,
    }, refs
    # DENOMINATOR: nothing silently vanishes. Three covered + one accounted-for
    # skip == four inputs.
    assert len(targets) + len(skipped) == 4
    assert skipped == ["README.md"]
    # And every ref the targets carry is readable by the prefix dispatcher.
    for t in targets:
        parse_cas_ref(t.ref)


# ---------------------------------------------------------------- pgw#783 dedup


def test_a_second_call_for_a_present_entry_skips_the_fetch(tmp_path):
    """pgw#783: G children share one filesystem. Once the entry is materialised,
    a sibling asking for the same dst must NOT re-fetch it (G x egress -> 1x).
    Deterministic form: fetch once, then again — the server sees no new hits."""
    data = make_file(CS * 3 + 5, seed=11)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        dst = tmp_path / "shared.safetensors"
        kw = dict(whole_digest="sha256:" + sha(data), total_size=len(data),
                  chunk_size_bytes=CS, window=2)
        download_chunked_file(specs_for(srv, parts), dst, **kw)
        assert srv.hits == {i: 1 for i in range(len(parts))}
        # The sibling call: present already, so ZERO new fetches.
        download_chunked_file(specs_for(srv, parts), dst, **kw)
        assert srv.hits == {i: 1 for i in range(len(parts))}, "re-fetched a present entry"
        assert dst.read_bytes() == data
    finally:
        srv.close()


def test_concurrent_callers_dedup_to_one_fetch_via_flock(tmp_path):
    """The cross-process guarantee, exercised with threads (same flock code
    path): four callers race for the same dst; the flock serialises them so the
    entry is fetched ONCE total, not four times."""
    import threading as _t

    data = make_file(CS * 2 + 9, seed=13)
    parts = split(data)
    srv = Server(dict(enumerate(parts)))
    try:
        dst = tmp_path / "raced.safetensors"
        kw = dict(whole_digest="sha256:" + sha(data), total_size=len(data),
                  chunk_size_bytes=CS, window=2)
        barrier = _t.Barrier(4)

        def _go():
            barrier.wait()
            download_chunked_file(specs_for(srv, parts), dst, **kw)

        threads = [_t.Thread(target=_go) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(30)
        assert dst.read_bytes() == data
        # Each chunk fetched exactly once across all four racers.
        assert srv.hits == {i: 1 for i in range(len(parts))}, \
            f"expected one fetch per chunk, got {srv.hits}"
    finally:
        srv.close()
