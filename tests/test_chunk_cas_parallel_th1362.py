"""Parallel POSITIONAL materialisation of a chunked CAS file.

Components arrive as multi-GB SINGLE files, so the reassembler is where parallel
transfer, resume and partial-failure re-fetch have to live. These drive the REAL
`download_chunked_file` over a real threaded HTTP server on localhost — real
sockets, real bodies, real threads, real files — because every property under
test is a property of the concurrency and the IO.

Two tiers:

*   Correctness (always on, kilobytes): byte-identical output vs a sequential
    reference assembly, digest-verified; resume from a partial; per-chunk
    refetch; out-of-order completion; failure leaves nothing installed.
*   Throughput (opt-in, multi-GB): ``CHUNK_CAS_BENCH=1 pytest -q -s
    tests/test_chunk_cas_parallel_th1362.py -k throughput``. The server caps
    PER-CONNECTION bandwidth, which is the regime that matters — an object
    store gives you ~100-250 MB/s on one ranged GET and much more in
    aggregate, so the only way to go faster is more concurrent chunks, and the
    only thing that bounds concurrency is RAM per in-flight chunk.

Run: pytest tests/test_chunk_cas_parallel_th1362.py -q
"""

from __future__ import annotations

import hashlib
import http.server
import multiprocessing
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

from gen_worker.models.chunk_cas import (
    ChunkSpec,
    DigestMismatch,
    chunk_len_at,
    download_chunked_file,
)

CS = 8192  # correctness-tier chunk size: the ARITHMETIC is what is under test


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stamp(base: bytes, index: int) -> bytes:
    """`base` with the chunk index stamped into its head, so every chunk of a
    synthetic file has a distinct sha256 while costing one memcpy to make."""
    return index.to_bytes(8, "big") + base[8:]


class _ChunkServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Enough handler threads that the server is never the bottleneck.
    block_on_close = False


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - silence the test server
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        try:
            idx = int(self.path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(404)
            return
        body = srv.body_for(idx)
        if body is None:
            self.send_error(404)
            return
        with srv.lock:
            srv.hits[idx] = srv.hits.get(idx, 0) + 1
            attempt = srv.hits[idx]
            srv.concurrent += 1
            srv.peak_concurrent = max(srv.peak_concurrent, srv.concurrent)
        try:
            mode = srv.modes.get(idx)
            if mode == "fail_once" and attempt == 1:
                self.close_connection = True
                return
            if mode == "corrupt_once" and attempt == 1:
                body = bytes([body[0] ^ 0xFF]) + body[1:]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_rate_limited(body, srv.rate_bps)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with srv.lock:
                srv.concurrent -= 1

    def _write_rate_limited(self, body: bytes, rate_bps: Optional[float]) -> None:
        if not rate_bps:
            self.wfile.write(body)
            return
        block = 1 << 20
        started = time.monotonic()
        sent = 0
        for off in range(0, len(body), block):
            self.wfile.write(body[off:off + block])
            sent += min(block, len(body) - off)
            owed = sent / rate_bps - (time.monotonic() - started)
            if owed > 0:
                time.sleep(owed)


def _start_server(
    blobs: Dict[int, bytes],
    *,
    base: Optional[bytes] = None,
    n_chunks: int = 0,
    last_len: int = 0,
    rate_bps: Optional[float] = None,
) -> _ChunkServer:
    """`blobs` serves explicit bodies (correctness tier); `base`+`n_chunks`
    synthesises them on demand (throughput tier — a 4 GiB corpus never has to
    exist anywhere at once)."""
    srv = _ChunkServer(("127.0.0.1", 0), _Handler)
    srv.lock = threading.Lock()
    srv.hits = {}
    srv.modes = {}
    srv.blobs = blobs
    srv.base = base
    srv.n_chunks = n_chunks
    srv.last_len = last_len
    srv.rate_bps = rate_bps
    srv.concurrent = 0
    srv.peak_concurrent = 0

    def body_for(idx: int) -> Optional[bytes]:
        if srv.blobs:
            return srv.blobs.get(idx)
        if srv.base is None or not (0 <= idx < srv.n_chunks):
            return None
        b = _stamp(srv.base, idx)
        if idx == srv.n_chunks - 1 and srv.last_len:
            b = b[: srv.last_len]
        return b

    srv.body_for = body_for
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _specs(srv: _ChunkServer, bodies: Sequence[bytes]) -> List[ChunkSpec]:
    host = f"http://127.0.0.1:{srv.server_address[1]}"
    return [
        ChunkSpec(sha256=_sha(b), url=f"{host}/chunk/{i}", length=len(b))
        for i, b in enumerate(bodies)
    ]


def _split(data: bytes, chunk_size: int) -> List[bytes]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def _sequential_reference(bodies: Sequence[bytes], dst: Path) -> str:
    """The assembly the old in-order commit loop performed, written out
    independently here so 'byte-identical to the sequential path' is an
    assertion against a REFERENCE and not against the code under test."""
    h = hashlib.sha256()
    with open(dst, "wb") as f:
        for b in bodies:
            f.write(b)
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total", [CS * 4, CS * 7 + 11, CS * 33 + 1])
def test_positional_assembly_is_byte_identical_to_sequential(tmp_path, total):
    data = os.urandom(total)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    try:
        ref = tmp_path / "ref.bin"
        whole = _sequential_reference(bodies, ref)

        got = tmp_path / "got.bin"
        download_chunked_file(
            _specs(srv, bodies), got,
            whole_digest="sha256:" + whole,
            total_size=total, chunk_size_bytes=CS, window=8,
        )
        assert got.read_bytes() == ref.read_bytes()
        assert _sha(got.read_bytes()) == whole
        # No debris: the tmp part file and its journal are gone. (The
        # cross-process `.casfetch.lock` is a permanent fixture, not debris.)
        left = sorted(p.name for p in tmp_path.iterdir()
                      if not p.name.endswith(".casfetch.lock"))
        assert left == ["got.bin", "ref.bin"]
    finally:
        srv.shutdown()


def test_window_wider_than_chunk_count_is_clamped(tmp_path):
    data = os.urandom(CS * 3)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    try:
        dst = tmp_path / "f.bin"
        download_chunked_file(
            _specs(srv, bodies), dst,
            whole_digest="sha256:" + _sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=64,
        )
        assert dst.read_bytes() == data
    finally:
        srv.shutdown()


def test_chunks_are_fetched_concurrently(tmp_path):
    """The whole point of item 1: a big single file must not materialise at one
    stream's speed. Proven by the server observing overlapping connections."""
    data = os.urandom(CS * 24)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)}, rate_bps=CS * 20)
    try:
        download_chunked_file(
            _specs(srv, bodies), tmp_path / "f.bin",
            whole_digest="sha256:" + _sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=8,
        )
        assert srv.peak_concurrent > 1, "assembly was single-streamed"
    finally:
        srv.shutdown()


def test_corrupt_chunk_is_refetched_and_the_region_is_rewritten(tmp_path):
    """A positionally-written chunk is verified AFTER it lands, so the retry has
    to overwrite its own region — the acceptance that byte-for-byte output is
    unaffected by a bad first attempt."""
    data = os.urandom(CS * 5)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    srv.modes[2] = "corrupt_once"
    try:
        dst = tmp_path / "f.bin"
        download_chunked_file(
            _specs(srv, bodies), dst,
            whole_digest="sha256:" + _sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=4,
        )
        assert dst.read_bytes() == data
        assert srv.hits[2] >= 2
    finally:
        srv.shutdown()


def test_dropped_connection_is_refetched(tmp_path):
    data = os.urandom(CS * 5)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    srv.modes[3] = "fail_once"
    try:
        dst = tmp_path / "f.bin"
        download_chunked_file(
            _specs(srv, bodies), dst,
            whole_digest="sha256:" + _sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=4,
        )
        assert dst.read_bytes() == data
        assert srv.hits[3] >= 2
    finally:
        srv.shutdown()


def test_lying_whole_file_digest_installs_nothing(tmp_path):
    """Fail-closed. Every chunk is individually valid, so this is the ONE place
    a manifest whose whole-file label disagrees with its own chunk list is
    caught — it must survive the move to positional writes."""
    data = os.urandom(CS * 4)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    try:
        dst = tmp_path / "f.bin"
        with pytest.raises(DigestMismatch):
            download_chunked_file(
                _specs(srv, bodies), dst,
                whole_digest="sha256:" + "b" * 64,
                total_size=len(data), chunk_size_bytes=CS, window=4,
            )
        assert not dst.exists()
    finally:
        srv.shutdown()


def test_resume_reuses_the_bytes_a_previous_run_committed(tmp_path):
    """Crash-resume: a run killed part-way must not re-fetch what it already
    made durable. Driven by killing the first attempt with an unservable chunk,
    then serving it — the second run's hit counts are the evidence."""
    data = os.urandom(CS * 8)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    dst = tmp_path / "f.bin"
    whole = "sha256:" + _sha(data)
    try:
        broken = _specs(srv, bodies)
        broken[6] = ChunkSpec(sha256="c" * 64, url=broken[6].url, length=CS)
        with pytest.raises(DigestMismatch):
            download_chunked_file(
                broken, dst, whole_digest=whole,
                total_size=len(data), chunk_size_bytes=CS, window=2,
            )
        assert not dst.exists()
        first_pass = dict(srv.hits)
        assert first_pass.get(0, 0) >= 1

        download_chunked_file(
            _specs(srv, bodies), dst, whole_digest=whole,
            total_size=len(data), chunk_size_bytes=CS, window=2,
        )
        assert dst.read_bytes() == data
        # Chunk 0 was durable before the failure, so the resume must not have
        # paid for it again.
        assert srv.hits[0] == first_pass[0], "resume re-fetched a durable chunk"
    finally:
        srv.shutdown()


def test_already_present_destination_is_not_refetched(tmp_path):
    data = os.urandom(CS * 3)
    bodies = _split(data, CS)
    srv = _start_server({i: b for i, b in enumerate(bodies)})
    try:
        dst = tmp_path / "f.bin"
        dst.write_bytes(data)
        download_chunked_file(
            _specs(srv, bodies), dst,
            whole_digest="sha256:" + _sha(data),
            total_size=len(data), chunk_size_bytes=CS, window=4,
        )
        assert srv.hits == {}
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------
# Throughput (opt-in)
# --------------------------------------------------------------------------

def _serve_in_subprocess(q, base: bytes, n_chunks: int, last_len: int,
                         rate_bps: float) -> None:  # pragma: no cover - child
    srv = _start_server({}, base=base, n_chunks=n_chunks, last_len=last_len,
                        rate_bps=rate_bps)
    q.put(srv.server_address[1])
    threading.Event().wait()


def _peak_rss_sampler(stop: threading.Event, out: List[int]) -> None:
    page = os.sysconf("SC_PAGE_SIZE")
    while not stop.is_set():
        try:
            with open("/proc/self/statm") as f:
                out.append(int(f.read().split()[1]) * page)
        except OSError:
            return
        stop.wait(0.02)


@pytest.mark.skipif(
    os.environ.get("CHUNK_CAS_BENCH") != "1",
    reason="multi-GB throughput measurement; set CHUNK_CAS_BENCH=1",
)
def test_throughput_multi_gb_synthetic(tmp_path):
    """A multi-GB single file over per-connection-capped links — the regime
    th#1362 pushes us into. Prints MB/s and peak RSS per window setting.

    Knobs: CHUNK_CAS_BENCH_GB (default 4), CHUNK_CAS_BENCH_CHUNK_MIB (64),
    CHUNK_CAS_BENCH_RATE_MBPS (per connection, 120),
    CHUNK_CAS_BENCH_WINDOWS (comma list, "6,16").
    """
    gib = float(os.environ.get("CHUNK_CAS_BENCH_GB", "4"))
    cs = int(os.environ.get("CHUNK_CAS_BENCH_CHUNK_MIB", "64")) << 20
    rate = float(os.environ.get("CHUNK_CAS_BENCH_RATE_MBPS", "120")) * 1e6
    windows = [int(w) for w in
               os.environ.get("CHUNK_CAS_BENCH_WINDOWS", "6,16").split(",")]

    total = int(gib * (1 << 30))
    n = (total + cs - 1) // cs
    last = total - cs * (n - 1)
    base = os.urandom(cs)

    # Whole-file and per-chunk digests, computed once over the synthetic corpus
    # without ever holding it all in RAM.
    whole_h = hashlib.sha256()
    digests: List[str] = []
    for i in range(n):
        b = _stamp(base, i)
        if i == n - 1:
            b = b[:last]
        digests.append(_sha(b))
        whole_h.update(b)
    whole = whole_h.hexdigest()

    # The server runs OUT OF PROCESS. In-process it shares the client's GIL,
    # which pins the measurement at roughly one connection's throughput no
    # matter how wide the window — a harness artifact that would have hidden
    # the effect entirely (measured: 171 MB/s at window 6 AND at window 16).
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    child = ctx.Process(target=_serve_in_subprocess,
                        args=(q, base, n, last, rate), daemon=True)
    child.start()
    port = q.get(timeout=30)
    host = f"http://127.0.0.1:{port}"
    specs = [
        ChunkSpec(sha256=digests[i], url=f"{host}/chunk/{i}",
                  length=chunk_len_at(total, i, cs))
        for i in range(n)
    ]
    try:
        print(f"\n[bench] {gib} GiB, {n} chunks of {cs >> 20} MiB, "
              f"per-connection cap {rate / 1e6:.0f} MB/s")
        for w in windows:
            dst = tmp_path / f"blob-w{w}.bin"
            samples: List[int] = []
            stop = threading.Event()
            sampler = threading.Thread(
                target=_peak_rss_sampler, args=(stop, samples), daemon=True)
            base_rss = int(open("/proc/self/statm").read().split()[1]) \
                * os.sysconf("SC_PAGE_SIZE")
            sampler.start()
            t0 = time.monotonic()
            download_chunked_file(
                specs, dst, whole_digest="sha256:" + whole,
                total_size=total, chunk_size_bytes=cs, window=w,
            )
            dt = time.monotonic() - t0
            stop.set()
            sampler.join(timeout=2)
            peak = max(samples) if samples else 0
            print(f"[bench] window={w:>3}  {total / dt / 1e6:8.1f} MB/s  "
                  f"{dt:6.1f}s  peak_rss=+{(peak - base_rss) >> 20} MiB")
            assert dst.stat().st_size == total
            dst.unlink()
    finally:
        child.terminate()
        child.join(timeout=5)
