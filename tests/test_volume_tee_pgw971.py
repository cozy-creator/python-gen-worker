"""pgw#971 / pgw#972: fill BOTH stores at once, and back off.

Three properties, all driven through the real `download_chunked_file` /
`ensure_snapshot_async` over a real threaded HTTP server on localhost — real
sockets, real bodies, real threads, real files. Every one of these is a
property of the concurrency and the IO, so a mock would assert nothing.

1.  **Tee.** With a volume attached, a chunked blob's bytes land in local CAS
    and on the volume in the SAME pass. The volume copy is byte-identical, is
    never observable under the digest name before the whole-file hash passes,
    and the post-fetch re-read (`mode=copy`) does not run for it.
2.  **Fused hash.** `_copy_verified_blob` still refuses bytes that do
    not hash to their ref — the hash moved into the copy loop, it did not go
    away.
3.  **Backoff.** A chunk source that delivers NOTHING gets an
    exponentially-backed-off retry; one that DELIVERED ITS FLOOR and then died
    is a lemon connection and gets a fresh socket with no sleep. Those are
    different failures and the loop must tell them apart.

Run: pytest tests/test_volume_tee_pgw971.py -q
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Dict, List

import pytest

import gen_worker.models.chunk_cas as cc
from gen_worker.models.chunk_cas import ChunkSpec, download_chunked_file
from gen_worker.models.cozy_snapshot import _copy_verified_blob, ensure_snapshot_async
from gen_worker.models.hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.refs import TensorhubRef

CS = 8192  # chunk size: the ARITHMETIC is what is under test, not the volume


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
# A real HTTP server serving chunks by digest, with per-digest fault injection.
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
            n = srv.hits[key]
            blob = srv.blobs.get(key)
            fail = srv.fail_first.get(key, 0)
            trickle = srv.trickle_bytes.get(key, 0)
        if blob is None:
            self.send_error(404)
            return
        if n <= fail:
            if trickle:
                # SOME bytes, under the floor, then the connection dies: the
                # lemon-connection shape.
                self.send_response(200)
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob[:trickle])
                self.wfile.flush()
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


@pytest.fixture()
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.blobs: Dict[str, bytes] = {}
    srv.hits: Dict[str, int] = {}
    srv.fail_first: Dict[str, int] = {}
    srv.trickle_bytes: Dict[str, int] = {}
    srv.lock = threading.Lock()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
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


# ---------------------------------------------------------------------------
# 1. The tee
# ---------------------------------------------------------------------------

def test_tee_fills_both_stores_in_one_pass(tmp_path: Path, server) -> None:
    payload = _body(CS * 9 + 17)
    specs = _publish(server, payload)
    local = tmp_path / "local" / "blob"
    volume = tmp_path / "volume" / "blob"

    published = download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_dst=volume,
    )

    assert published is True
    assert local.read_bytes() == payload
    assert volume.read_bytes() == payload
    # No staging litter left behind on either side.
    assert [p.name for p in volume.parent.iterdir()] == ["blob"]


def test_no_mirror_requested_publishes_nothing(tmp_path: Path, server) -> None:
    payload = _body(CS * 3)
    specs = _publish(server, payload)
    local = tmp_path / "local" / "blob"

    published = download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=4,
    )
    assert published is False
    assert local.read_bytes() == payload


def test_failed_download_leaves_no_volume_blob(tmp_path: Path, server) -> None:
    """A half-written volume blob must never be readable under the digest
    name. The mirror stages writer-unique and is published only after the
    LOCAL file proves its whole-file hash."""
    payload = _body(CS * 4)
    specs = _publish(server, payload)
    # A lying whole-file label: every chunk is valid, the file is not.
    local = tmp_path / "local" / "blob"
    volume = tmp_path / "volume" / "blob"

    with pytest.raises(cc.DigestMismatch):
        download_chunked_file(
            specs, local, whole_digest="sha256:" + ("b7" * 32),
            total_size=len(payload), chunk_size_bytes=CS, window=4,
            mirror_dst=volume,
        )

    assert not local.exists()
    assert not volume.exists()
    assert list(volume.parent.iterdir()) == []  # no orphaned stage file either


def test_mirror_failure_never_fails_the_download(tmp_path: Path, server) -> None:
    """A volume that cannot be written to costs the request nothing."""
    payload = _body(CS * 3)
    specs = _publish(server, payload)
    local = tmp_path / "local" / "blob"
    # The mirror's parent directory path is occupied by a FILE, so mkdir fails.
    blocker = tmp_path / "volume"
    blocker.write_bytes(b"not a directory")
    volume = blocker / "aa" / "blob"

    published = download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_dst=volume,
    )
    assert published is False
    assert local.read_bytes() == payload  # the fetch itself is untouched


def test_resume_seeds_the_mirror_from_local_disk(tmp_path: Path, server) -> None:
    """Chunks adopted from a previous run are never refetched, so the mirror
    takes them off local disk — otherwise it would publish a file with holes."""
    payload = _body(CS * 6)
    specs = _publish(server, payload)
    local = tmp_path / "local" / "blob"
    volume = tmp_path / "volume" / "blob"
    local.parent.mkdir(parents=True, exist_ok=True)

    # A prior run that wrote and journalled the first three chunks.
    part = local.parent / f".{local.name}.chunkpart"
    with open(part, "wb") as f:
        f.write(payload[: CS * 3])
        f.write(b"\0" * (len(payload) - CS * 3))
    journal = local.parent / f".{local.name}.chunkdone"
    journal.write_text(
        f"sha256:{_sha(payload)} {len(payload)}\n0\n1\n2\n"
    )
    before = dict(server.hits)

    published = download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=4,
        mirror_dst=volume,
    )

    assert published is True
    assert volume.read_bytes() == payload
    # The adopted chunks really were adopted, not refetched.
    for spec in specs[:3]:
        assert server.hits.get(spec.sha256, 0) == before.get(spec.sha256, 0)


# ---------------------------------------------------------------------------
# ensure_snapshot_async: which write-through path each blob kind takes
# ---------------------------------------------------------------------------

def _resolved(payload: bytes, specs: List[ChunkSpec], small: bytes, small_url: str):
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
            WorkerResolvedRepoFile(
                path="config.json",
                size_bytes=len(small),
                digest="sha256:" + _sha(small),
                url=small_url,
            ),
        ],
    )


def test_chunked_blob_tees_and_whole_file_blob_copies(
    tmp_path: Path, server, caplog,
) -> None:
    payload = _body(CS * 5, seed=3)
    specs = _publish(server, payload)
    small = b'{"k":1}'
    with server.lock:
        server.blobs[_sha(small)] = small
    small_url = f"http://127.0.0.1:{server.server_address[1]}/chunk/{_sha(small)}"

    local = tmp_path / "local"
    volume = tmp_path / "volume"
    caplog.set_level("INFO", logger="gen_worker.models.chunk_cas")
    caplog.set_level("INFO", logger="gen_worker.download")

    snap = asyncio.run(ensure_snapshot_async(
        base_dir=local,
        ref=TensorhubRef(owner="org", repo="model"),
        resolved=_resolved(payload, specs, small, small_url),
        fill_source_dir=volume,
    ))

    assert (snap / "model.safetensors").read_bytes() == payload
    assert (snap / "config.json").read_bytes() == small
    # Both blobs warmed the volume, by the path each blob kind should take.
    modes = {
        line.split("digest=")[0].split("mode=")[1].strip()
        for line in caplog.text.splitlines() if "blob_fill_publish" in line
    }
    assert modes == {"tee", "copy"}
    vol_blobs = sorted(p.name for p in (volume / "blobs" / "sha256").rglob("*")
                       if p.is_file())
    assert vol_blobs == sorted([_sha(payload), _sha(small)])


# ---------------------------------------------------------------------------
# 2. pgw#971: the hash is fused into the copy, not deleted from it
# ---------------------------------------------------------------------------

def test_fused_hash_still_refuses_a_wrong_digest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    payload = _body(1 << 16)
    src.write_bytes(payload)
    # Right size, wrong content — a size-only check would wave this through.
    ok = _copy_verified_blob(src, dst, "sha256:" + ("de" * 32), len(payload))
    assert ok is False
    assert not dst.exists()
    assert list(tmp_path.iterdir()) == [src]  # stage file cleaned up


def test_fused_hash_accepts_honest_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "sub" / "dst"
    payload = _body(1 << 16, seed=5)
    src.write_bytes(payload)
    assert _copy_verified_blob(src, dst, "sha256:" + _sha(payload), len(payload))
    assert dst.read_bytes() == payload


# ---------------------------------------------------------------------------
# 3. pgw#972: which retry branch each failure shape takes
#
# These assert the DECISION, never the runner's speed: the loop logs
# the classification it made and the delay it scheduled, so the branch taken is
# the work's product and is observable without a clock. Timing the download and
# comparing to a constant would measure the CI runner, and would go red on a
# loaded box while the policy was perfectly correct.
# ---------------------------------------------------------------------------

def _decisions(caplog, sha256: str):
    """(route_alive, retry_in) for each retry decision about one chunk."""
    out = []
    for line in caplog.text.splitlines():
        if "chunk_refetch" not in line or sha256[:12] not in line:
            continue
        alive = line.split("route_alive=")[1].split()[0] == "True"
        delay = float(line.split("retry_in=")[1].split("s")[0])
        out.append((alive, delay))
    return out


def test_a_route_that_delivers_nothing_is_made_to_wait(
    tmp_path: Path, server, monkeypatch, caplog,
) -> None:
    """The defect: a chunk source that delivers NOTHING was retried with no
    delay at all, so a transient TLS/connect blip burned the whole budget in
    well under a second. Each such failure must now schedule a real delay, and
    the delays must ESCALATE."""
    monkeypatch.setattr(cc, "_CHUNK_BACKOFF_CAP_S", 0.5)
    caplog.set_level("WARNING", logger="gen_worker.models.chunk_cas")
    payload = _body(CS * 2)
    specs = _publish(server, payload)
    with server.lock:
        server.fail_first[specs[0].sha256] = 3  # three connect-shaped failures

    local = tmp_path / "local" / "blob"
    download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=2,
    )

    assert local.read_bytes() == payload  # it still succeeds
    decisions = _decisions(caplog, specs[0].sha256)
    assert len(decisions) == 3
    assert all(not alive for alive, _ in decisions), decisions
    assert all(delay > 0 for _, delay in decisions), decisions
    # Each scheduled delay lies in the window its own strike defines. The bound
    # is DERIVED from the schedule, not a guess about how fast the box is.
    cap = cc._CHUNK_BACKOFF_CAP_S
    for strike, (_, delay) in enumerate(decisions, start=1):
        window = min(cap, 2.0 ** strike)
        assert 0.5 * window <= delay <= window, (strike, delay, decisions)


def test_a_live_route_reconnects_at_once(
    tmp_path: Path, server, monkeypatch, caplog,
) -> None:
    """pgw#786's rule survives: an attempt that DELIVERED ITS FLOOR and then
    died is a lemon CONNECTION, and the answer is a fresh socket NOW, not a
    sleep. The read block and the floor are shrunk so the property is reachable
    in kilobytes; the arithmetic under test is unchanged."""
    monkeypatch.setattr(cc, "_CHUNK_BACKOFF_CAP_S", 30.0)
    monkeypatch.setattr(cc, "_READ_CHUNK_BYTES", 64)
    monkeypatch.setattr(cc, "_CHUNK_PROGRESS_FLOOR_BYTES", 128)
    caplog.set_level("WARNING", logger="gen_worker.models.chunk_cas")
    payload = _body(CS * 2)
    specs = _publish(server, payload)
    with server.lock:
        server.fail_first[specs[0].sha256] = 3
        server.trickle_bytes[specs[0].sha256] = 4096  # >> the shrunken floor

    local = tmp_path / "local" / "blob"
    download_chunked_file(
        specs, local, whole_digest="sha256:" + _sha(payload),
        total_size=len(payload), chunk_size_bytes=CS, window=2,
    )

    assert local.read_bytes() == payload
    decisions = _decisions(caplog, specs[0].sha256)
    assert decisions, "no retry was recorded"
    assert all(alive for alive, _ in decisions), decisions
    assert all(delay == 0.0 for _, delay in decisions), decisions


def test_a_digest_mismatch_refetches_at_once(
    tmp_path: Path, server, monkeypatch, caplog,
) -> None:
    """A complete, wrong-hashing body proves the ROUTE is healthy and only the
    bytes are not. Backing off there is pure loss — and it made an existing
    permanently-bad-chunk test walk the whole ladder instead of failing fast."""
    monkeypatch.setattr(cc, "_CHUNK_BACKOFF_CAP_S", 30.0)
    caplog.set_level("WARNING", logger="gen_worker.models.chunk_cas")
    payload = _body(CS * 2)
    specs = _publish(server, payload)
    # Serve chunk 0 under a digest it does not hash to: every fetch is complete
    # and every fetch mismatches.
    with server.lock:
        server.blobs[specs[0].sha256] = _body(CS, seed=99)

    local = tmp_path / "local" / "blob"
    with pytest.raises(Exception):
        download_chunked_file(
            specs, local, whole_digest="sha256:" + _sha(payload),
            total_size=len(payload), chunk_size_bytes=CS, window=2,
        )

    decisions = _decisions(caplog, specs[0].sha256)
    assert decisions, "no retry was recorded"
    assert all(alive for alive, _ in decisions), decisions
    assert all(delay == 0.0 for _, delay in decisions), decisions


def test_a_dead_route_gives_up_on_a_COUNT_not_a_clock(
    tmp_path: Path, server, monkeypatch,
) -> None:
    """The give-up is `_CHUNK_MAX_ATTEMPTS` attempts, per `stall.py`'s standing
    rule that nothing which ends real work may key on a wall clock. The
    evidence is the SERVER's hit count, which no amount of box load can move."""
    monkeypatch.setattr(cc, "_CHUNK_BACKOFF_CAP_S", 0.05)
    payload = _body(CS * 2)
    specs = _publish(server, payload)
    with server.lock:
        server.fail_first[specs[0].sha256] = 10_000

    local = tmp_path / "local" / "blob"
    with pytest.raises(Exception):
        download_chunked_file(
            specs, local, whole_digest="sha256:" + _sha(payload),
            total_size=len(payload), chunk_size_bytes=CS, window=2,
        )
    assert not local.exists()
    assert server.hits[specs[0].sha256] == cc._CHUNK_MAX_ATTEMPTS
